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
Functionalities for transformations of raster objects.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING,  Any, Literal, TypeVar, Callable

import affine
import rasterio as rio
from rasterio.crs import CRS
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import pandas as pd
from packaging.version import Version

from geoutils import profiler
from geoutils._dispatch import _check_match_bbox, _check_match_grid
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum
from geoutils.projtools import _get_bounds_projected, _get_footprint_projected
from geoutils.raster.referencing import (
    _default_nodata,
    _res,
)
from geoutils._misc import silence_rasterio_message, import_optional
from geoutils.multiproc.mparray import MultiprocConfig, _write_multiproc_result

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike, RasterType
    from geoutils.raster.raster import Raster
    from geoutils.vector.vector import VectorLike

# Dask as optional dependency
try:
    import dask
    import dask.array as da
    from dask import delayed
    from dask.utils import cached_cumsum
except ImportError:

    def delayed(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake delayed decorator if dask is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

##############
# 1/ REPROJECT
##############

# SUBFUNCTIONS

def _resampling_method_from_str(method_str: str) -> rio.enums.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.enums.Resampling:
        if method.name == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.enums.Resampling method. "
            f"Valid methods: {[method.name for method in rio.enums.Resampling]}"
        )
    return resampling_method


def _check_reproj_nodata_dtype(
    source_raster: RasterType,
    nodata: int | float | None,
    dtype: DTypeLike | None,
    force_source_nodata: int | float | None,
) -> tuple[DTypeLike, int | float | None, int | float | None]:
    """Check user inputs of reproject regarding nodata and data type."""

    # Set output dtype
    if dtype is None:
        # Warning: this will not work for multiple bands with different dtypes
        dtype = source_raster.dtype

    # --- Set source nodata if provided -- #
    if force_source_nodata is None:
        src_nodata = source_raster.nodata
    else:
        src_nodata = force_source_nodata
        # Raise warning if a different nodata value exists for this raster than the forced one (not None)
        if source_raster.nodata is not None:
            warnings.warn(
                "Forcing source nodata value of {} despite an existing nodata value of {} in the raster. "
                "To silence this warning, use self.set_nodata() before reprojection instead of forcing.".format(
                    force_source_nodata, source_raster.nodata
                )
            )

    # --- Set destination nodata if provided -- #
    # This is needed in areas not covered by the input data.
    # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
    if nodata is None:
        nodata = source_raster.nodata
        if nodata is None:
            nodata = _default_nodata(dtype)
            # If nodata is already being used, raise a warning.
            if not source_raster.is_loaded:
                warnings.warn(
                    f"For reprojection, nodata must be set. Setting default nodata to {nodata}. You may "
                    f"set a different nodata with `nodata`."
                )

            elif nodata in source_raster.data:
                warnings.warn(
                    f"For reprojection, nodata must be set. Default chosen value {nodata} exists in "
                    f"self.data. This may have unexpected consequences. Consider setting a different nodata with "
                    f"self.set_nodata()."
                )

    return dtype, src_nodata, nodata


def _is_reproj_needed(src_shape: tuple[int, int], reproj_kwargs: dict[str, Any]) -> bool:
    """Check if reprojection is actually needed based on transformation parameters."""

    src_transform = reproj_kwargs["src_transform"]
    transform = reproj_kwargs["dst_transform"]
    src_crs = reproj_kwargs["src_crs"]
    crs = reproj_kwargs["dst_crs"]
    grid_size = reproj_kwargs["dst_shape"][::-1]
    src_res = _res(src_transform)
    res = _res(transform)

    # Caution, grid_size is (width, height) while shape is (height, width)
    return all(
        [
            (transform == src_transform) or (transform is None),
            (crs == src_crs) or (crs is None),
            (grid_size == src_shape[::-1]) or (grid_size is None),
            np.all(np.array(res) == src_res) or (res is None),
        ]
    )


def _rio_reproject(src_arr: NDArrayNum, reproj_kwargs: dict[str, Any]) -> NDArrayNum:
    """Rasterio reprojection wrapper.

    :param src_arr: Source array for data.
    :param reproj_kwargs: Reprojection parameter dictionary.
    """

    # All masked values must be set to a nodata value for rasterio's reproject to work properly
    if np.ma.isMaskedArray(src_arr):
        is_input_masked = True
        src_mask = np.ma.getmaskarray(src_arr)
        src_arr = src_arr.data  # type: ignore
    else:
        is_input_masked = False
        src_mask = ~np.isfinite(src_arr)

    # Check reprojection is possible with nodata (boolean raster will be converted, so no need to check)
    if np.dtype(src_arr.dtype) != bool and (reproj_kwargs["src_nodata"] is None and np.sum(src_mask) > 0):
        raise ValueError(
            "No nodata set, set one for the raster with self.set_nodata() or use a temporary one "
            "with `force_source_nodata`."
        )

    # For a boolean type
    convert_bool = False
    if np.dtype(src_arr.dtype) == np.bool_:
        # To convert back later
        convert_bool = True
        # Convert to uint8 for nearest, float otherwise
        if reproj_kwargs["resampling"] in [Resampling.nearest, "nearest"]:
            src_arr = src_arr.astype("uint8")  # type: ignore
        else:
            warnings.warn(
                "Reprojecting a raster mask (boolean type) with a resampling method other than 'nearest', "
                "results in the boolean array being converted to float during reprojection."
            )
            src_arr = src_arr.astype("float32")  # type: ignore

        # Convert automated output dtype to the input dtype
        if np.dtype(reproj_kwargs["dtype"]) == np.bool_:
            reproj_kwargs["dtype"] = src_arr.dtype

        # Update nodata value, which won't exist
        reproj_kwargs["src_nodata"] = _default_nodata(src_arr.dtype)

    # Fill with nodata values on mask
    if reproj_kwargs["src_nodata"] is not None:
        src_arr[src_mask] = reproj_kwargs["src_nodata"]

    # Check if multiband
    is_multiband = len(src_arr.shape) > 2

    # Prepare destination array
    shape = (src_arr.shape[0], *reproj_kwargs["dst_shape"]) if is_multiband else reproj_kwargs["dst_shape"]
    dst_arr = np.zeros(shape, dtype=reproj_kwargs["dtype"])

    # Performance keywords
    if reproj_kwargs["num_threads"] == 0:
        # Default to cpu count minus one. If the cpu count is undefined, num_threads will be 1
        cpu_count = os.cpu_count() or 2
        num_threads = cpu_count - 1
    else:
        num_threads = reproj_kwargs["num_threads"]

    # We force XSCALE=1 and YSCALE=1 passed to GDAL.Warp to avoid resampling deformations depending on extent/shape,
    # which leads to different results on chunks or a full array
    # See: https://gdal.org/en/stable/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions
    # And: https://github.com/rasterio/rasterio/issues/2995
    reproj_kwargs.update(
        {
            "num_threads": num_threads,
            "XSCALE": 1,
            "YSCALE": 1,
        }
    )
    # If Rasterio is recent enough version, force tolerance to 0 to avoid deformations on chunks
    # See: https://github.com/rasterio/rasterio/issues/2433#issuecomment-2786157846
    if Version(rio.__version__) >= Version("1.5.0"):
        reproj_kwargs.update({"tolerance": 0})

    # Pop dtype and dst_shape arguments that don't exist in Rasterio, and are only used above
    reproj_kwargs.pop("dtype")
    reproj_kwargs.pop("dst_shape")

    # Rasterio raises a warning that src_transform are not defined when multiple ones are passed during chunked ops
    # (Dask/Multiproc), although this is not the case, maybe a bug upstream?
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

    # XSCALE/YSCALE have been supported for a while, but not officially exposed in the API until Rasterio 1.5,
    # so we need to silence them in warnings to avoid noise for users
    with silence_rasterio_message(param_name="SCALE"):
        # Run reprojection
        _ = rio.warp.reproject(src_arr, dst_arr, **reproj_kwargs)

    # Get output mask
    if reproj_kwargs["dst_nodata"] is not None:
        dst_mask = dst_arr == reproj_kwargs["dst_nodata"]
    else:
        dst_mask = np.zeros(dst_arr.shape, dtype=bool)

    # If output needs to be converted back to boolean
    if convert_bool:
        dst_arr = dst_arr.astype(bool)

    # Set mask
    if is_input_masked:
        dst_arr = np.ma.masked_array(data=dst_arr, mask=dst_mask, fill_value=reproj_kwargs["dst_nodata"])
    else:
        dst_arr[dst_mask] = np.nan

    return dst_arr


# CHUNKED LOGIC (for both Dask and Multiprocessing)

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
    src_count: int,
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
    # Raised warning for buffer is not important
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Geometry is in a geographic CRS.*")
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
        c.update({"dst_shape": dst_block_geogrids[i].shape, "dst_transform": tuple(dst_block_geogrids[i].transform),
                  "dst_count": src_count})

    return src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids


def _reproject_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """

    is_multiband = combined_meta["dst_count"] >= 2

    # If no source chunk intersects, we return a chunk of destination nodata values
    if len(src_arrs) == 0:
        # We can use float32 to return NaN, will be cast to other floating type later if that's not source array dtype
        dst_shape = (combined_meta["dst_count"], *combined_meta["dst_shape"]) if is_multiband \
            else combined_meta["dst_shape"]
        dst_arr = np.zeros(dst_shape, dtype=np.dtype("float32"))
        dst_arr[:] = np.nan
        return dst_arr

    # First, we build an empty array with the combined shape, only with nodata values
    shape = (src_arrs[0].shape[0], *combined_meta["src_shape"]) if is_multiband else combined_meta["src_shape"]

    comb_src_arr = np.full(shape, kwargs["src_nodata"], dtype=src_arrs[0].dtype)
    if np.ma.isMaskedArray(src_arrs[0]):
        comb_src_arr = np.ma.masked_array(data=comb_src_arr)

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
            "num_threads": 1,
        }
    )
    # Define dtype if undefined
    if "dtype" not in kwargs:
        kwargs.update({"dtype": comb_src_arr.dtype})

    dst_arr = _rio_reproject(src_arr=comb_src_arr, reproj_kwargs=kwargs)  # type: ignore

    return dst_arr


@delayed
def _delayed_reproject_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Delayed reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """
    return _reproject_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)


def _dask_reproject(
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

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    import time
    t0 = time.time()
    # Define the chunking
    # For source, we can use the .chunks attribute
    src_chunks = darr.chunks[-2:]  # In case input is multi-band

    if dst_chunksizes is None:
        dst_chunksizes = (darr.chunksize[-2], darr.chunksize[-1])  # In case input is multi-band

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta(
            src_count=darr.shape[0] if darr.ndim == 3 else 1,
            src_shape=darr.shape[-2:],  # In case input is multi-band
            src_transform=src_transform,
            src_crs=src_crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_chunks=src_chunks,
            dst_chunksizes=dst_chunksizes,
        )
    )

    t1 = time.time()
    print(t1 - t0)

    # We call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block

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
    blocks_delayed = darr.to_delayed()

    # Spatial block grid shape (from spatial chunks)
    is_multiband = darr.ndim == 3
    ny_src = len(src_chunks[0])
    nx_src = len(src_chunks[1])
    src_yi, src_xi = np.unravel_index(np.arange(ny_src * nx_src), shape=(ny_src, nx_src))
    # Normalize band groups:
    # - 2D: one pseudo group (bb=None, nb=0)
    # - 3D: real band blocks with their sizes
    band_groups: list[tuple[int | None, int]] = (
        [(None, 0)] if not is_multiband else [(bb, int(sz)) for bb, sz in enumerate(darr.chunks[0])]
    )
    # Output data type
    out_dtype = np.dtype(kwargs.get("dtype", darr.dtype))

    # Helper function to support both 2D and 3D cases
    def _dst_block_as_da(i: int) -> da.Array:
        """Build destination block as a Dask array (2D or 3D)."""
        shp2 = dst_block_geogrids[i].shape  # (ydst, xdst)

        # Spatial source coords for this destination tile
        coords = [(src_yi[j], src_xi[j]) for j in dest2source[i]]

        def _src_chunks_for_group(bb: int | None) -> list[Any]:
            # Accounting for the fact that blocks_delayed is either (ny,nx) or (nb,ny,nx)
            if bb is None:
                return [blocks_delayed[y, x] for (y, x) in coords]
            return [blocks_delayed[bb, y, x] for (y, x) in coords]

        def _one_group(bb: int | None, nb: int) -> da.Array:
            r = _delayed_reproject_per_block(
                *_src_chunks_for_group(bb),
                block_ids=meta_params[i][1],
                combined_meta=meta_params[i][0],
                **kwargs,
            )
            shape = shp2 if bb is None else (nb, *shp2)
            # We define the expected output shape and dtype to simplify things for Dask
            return da.from_delayed(r, shape=shape, dtype=out_dtype)

        # Build per-group outputs then concatenate along band axis if needed
        groups = [_one_group(bb, nb) for (bb, nb) in band_groups]
        return groups[0] if len(groups) == 1 else da.concatenate(groups, axis=0)

    # Run the delayed reprojection, looping for each destination block-band (2D block and 1D band-chunk)
    list_reproj_da = [_dst_block_as_da(i) for i in range(len(dest2source))]

    # Array comes out as flat blocks x chunksize0 (varying) x chunksize1 (varying), so we can't reshape directly
    # We need to unravel the flattened blocks indices to align X/Y, then concatenate all columns, then rows
    ny_dst, nx_dst = len(dst_chunks[0]), len(dst_chunks[1])
    iy, ix = np.unravel_index(np.arange(len(dest2source)), shape=(ny_dst, nx_dst))
    ax_x = 1 if darr.ndim == 2 else 2  # Adjust axes depending on if raster is single-band or multi-band
    ax_y = 0 if darr.ndim == 2 else 1
    rows = [
        da.concatenate([list_reproj_da[k] for k in range(len(list_reproj_da)) if iy[k] == r], axis=ax_x)
        for r in range(ny_dst)
    ]
    concat_all = da.concatenate(rows, axis=ax_y)
    return concat_all

def _wrapper_multiproc_reproject_per_block(
    rst: Raster,
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
    rst: Raster,
    mp_config: MultiprocConfig,
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
    src_chunks = _chunks2d_from_chunksizes_shape(chunksizes=(mp_config.chunk_size, mp_config.chunk_size),
                                                 shape=rst.shape)
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta(
            src_count=rst.count,
            src_shape=rst.shape,
            src_transform=rst.transform,
            src_crs=rst.crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_chunks=src_chunks,
            dst_chunksizes=(mp_config.chunk_size, mp_config.chunk_size),
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
            mp_config.cluster.launch_task(
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
    _write_multiproc_result(tasks, mp_config, file_metadata)


@profiler.profile("geoutils.raster.geotransformations._reproject", memprof=True)
def _reproject(
    source_raster: RasterType,
    ref: RasterLike,
    crs: CRS | str | int | None = None,
    res: float | tuple[float, float] | None = None,
    grid_size: tuple[int, int] | None = None,
    bounds: tuple[float, float, float, float] | rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    force_source_nodata: int | float | None = None,
    silent: bool = False,
    n_threads: int = 0,
    memory_limit: int = 64,
    mp_config: MultiprocConfig | None = None,
) -> tuple[bool, NDArrayNum | NDArrayBool | None, affine.Affine | None, CRS | None, int | float | None]:
    """
    Reproject raster. See Raster.reproject() for details.
    """

    # 1/ Check and normalize match-grid inputs
    dst_shape, dst_transform, dst_crs = _check_match_grid(
        src=source_raster, ref=ref, res=res, shape=grid_size, bounds=bounds, crs=crs, coords=None
    )

    # 2/ Check user input for nodata and dtype
    dtype, src_nodata, nodata = _check_reproj_nodata_dtype(
        source_raster=source_raster,
        nodata=nodata,
        dtype=dtype,
        force_source_nodata=force_source_nodata,
    )

    # 3/ Store georeferencing parameters for reprojection
    reproj_kwargs = {
        "src_transform": source_raster.transform,
        "dst_transform": dst_transform,
        "src_crs": source_raster.crs,
        "dst_crs": dst_crs,
        "resampling": resampling if isinstance(resampling, Resampling) else _resampling_method_from_str(resampling),
        "src_nodata": src_nodata,
        "dst_nodata": nodata,
        "dtype": dtype,
        "dst_shape": dst_shape,
    }

    # 4/ Check if reprojection is needed, otherwise return source raster with warning
    if _is_reproj_needed(src_shape=source_raster.shape, reproj_kwargs=reproj_kwargs):
        if (nodata == src_nodata) or (nodata is None):
            if not silent:
                warnings.warn("Output projection, bounds and grid size are identical -> returning self (not a copy!)")
            return True, None, None, None, None

        elif nodata is not None:
            if not silent:
                warnings.warn(
                    "Only nodata is different, consider using the 'set_nodata()' method instead'\
                ' -> returning self (not a copy!)"
                )
            return True, None, None, None, None

    # 5/ Perform reprojection
    reproj_kwargs.update({"num_threads": n_threads, "warp_mem_limit": memory_limit})
    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from reproject(). To use Multiprocessing, open the file without chunks."
        )

    # If using Multiprocessing backend, process and return None (files written on disk)
    if mp_config is not None:
        _multiproc_reproject(source_raster, mp_config=mp_config, **reproj_kwargs)  # type: ignore
        return False, None, None, None, None

    # If using Dask backend, process and return Dask array
    if da is not None and isinstance(source_raster.data, da.Array):
        dst_arr = _dask_reproject(darr=source_raster.data, **reproj_kwargs)

    # If using direct reprojection, process and return NumPy array
    else:
        dst_arr = _rio_reproject(src_arr=source_raster.data, reproj_kwargs=reproj_kwargs)

    return False, dst_arr, reproj_kwargs["dst_transform"], reproj_kwargs["dst_crs"], reproj_kwargs["dst_nodata"]


#########
# 2/ CROP
#########

@profiler.profile("geoutils.raster.geotransformations._crop", memprof=True)
def _crop(
    source_raster: RasterType,
    bbox: RasterLike | VectorLike | tuple[float, float, float, float],
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> tuple[NDArrayNum, affine.Affine]:
    """Crop raster. See details in Raster.crop()."""

    # Check input, raise appropriate errors and warnings
    bbox = _check_match_bbox(source_raster, bbox)

    assert distance_unit in ["georeferenced", "pixel"], "distance_unit must be 'georeferenced' or 'pixel'"

    # If using georeferenced unit, use bbox directly
    if distance_unit == "georeferenced":
        xmin, ymin, xmax, ymax = bbox
    # Else, convert to ij
    else:
        colmin, rowmin, colmax, rowmax = bbox
        xmin, ymax = rio.transform.xy(source_raster.transform, rowmin, colmin, offset="ul")
        xmax, ymin = rio.transform.xy(source_raster.transform, rowmax, colmax, offset="ul")

    # Finding the intersection of requested bounds and original bounds, cropped to image shape
    ref_win = rio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=source_raster.transform)
    self_win = rio.windows.from_bounds(*source_raster.bounds, transform=source_raster.transform).crop(
        *source_raster.shape
    )
    final_window = ref_win.intersection(self_win).round_lengths().round_offsets()

    # Update bounds and transform accordingly
    new_xmin, new_ymin, new_xmax, new_ymax = rio.windows.bounds(final_window, transform=source_raster.transform)
    tfm = rio.transform.from_origin(new_xmin, new_ymax, *source_raster.res)

    if source_raster._is_xr:

        (rowmin, rowmax), (colmin, colmax) = final_window.toranges()
        assert source_raster._obj is not None
        crop_img = source_raster._obj.isel(y=slice(rowmin, rowmax), x=slice(colmin, colmax))

    elif source_raster.is_loaded:
        # In case data is loaded on disk, can extract directly from np array
        (rowmin, rowmax), (colmin, colmax) = final_window.toranges()
        crop_img = source_raster.data[..., rowmin:rowmax, colmin:colmax]

    else:

        assert source_raster._disk_shape is not None  # This should not be the case, sanity check to make mypy happy

        # If data was not loaded, and self's transform was updated (e.g. due to downsampling) need to
        # get the Window corresponding to on disk data
        ref_win_disk = rio.windows.from_bounds(
            new_xmin, new_ymin, new_xmax, new_ymax, transform=source_raster._disk_transform
        )
        self_win_disk = rio.windows.from_bounds(*source_raster.bounds, transform=source_raster._disk_transform).crop(
            *source_raster._disk_shape[1:]
        )
        final_window_disk = ref_win_disk.intersection(self_win_disk).round_lengths().round_offsets()

        # Round up to downsampling size, to match __init__
        final_window_disk = rio.windows.round_window_to_full_blocks(
            final_window_disk, ((source_raster._downsample, source_raster._downsample),)
        )

        # Load data for "on_disk" window but out_shape matching in-memory transform -> enforce downsampling
        # AD (24/04/24): Note that the same issue as #447 occurs here when final_window_disk extends beyond
        # self's bounds. Using option `boundless=True` solves the issue but causes other tests to fail
        # This should be fixed with #447 and previous line would be obsolete.
        with rio.open(source_raster.name) as raster:
            crop_img = raster.read(
                indexes=source_raster._bands,
                masked=source_raster._masked,
                window=final_window_disk,
                out_shape=(final_window.height, final_window.width),
            )

        # Squeeze first axis for single-band
        if crop_img.ndim == 3 and crop_img.shape[0] == 1:
            crop_img = crop_img.squeeze(axis=0)

    return crop_img, tfm


##############
# 3/ TRANSLATE
##############


@profiler.profile("geoutils.raster.geotransformations._translate", memprof=True)
def _translate(
    transform: affine.Affine,
    xoff: float,
    yoff: float,
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> affine.Affine:
    """
    Translate geotransform horizontally, either in pixels or georeferenced units.

    :param transform: Input geotransform.
    :param xoff: Translation x offset.
    :param yoff: Translation y offset.
    :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.

    :return: Translated transform.
    """

    if distance_unit not in ["georeferenced", "pixel"]:
        raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

    # Get transform
    dx, b, xmin, d, dy, ymax = list(transform)[:6]

    # Convert pixel offsets to georeferenced units
    if distance_unit == "pixel":
        xoff *= dx
        yoff *= dy

    return rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)
