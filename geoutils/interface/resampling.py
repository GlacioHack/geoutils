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

"""Functionalities for resampling a regular raster grid at points (uses interpolator/reducers in operators)."""

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Literal, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio

from geoutils._config import config
from geoutils._dispatch import _check_match_points, _get_pointcloud_interface, is_dask_geodataframe
from geoutils._misc import import_optional
from geoutils._typing import DTypeLike, NDArrayBool, NDArrayNum, Number
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.chunked import cached_cumsum, normalize_chunks
from geoutils.multiproc.mparray import block_bounds_from_chunks
from geoutils.operators.execution import _resample_at_points as _resample_array_at_points
from geoutils.operators.interpolator import (
    Interpolator,
    RasterConvolution,
    RegularInterpolationMethod,
    ScipyInterpolator,
    _interp_output_dtype,
    _interpolate_array_band,
    _regular_interpolation_method,
    _resolve_interpolator,
)
from geoutils.operators.neighbours import (
    GridCoverage,
    GridNeighbours,
    PointNeighbours,
    _check_regular_grid_neighbours,
    _compute_resampling_overlap,
    _configure_grid_neighbours,
    _prepare_regular_interpolation_data,
    _resolve_grid_neighbours_for_interpolator,
)
from geoutils.operators.nodata import (
    NodataChoice,
    NodataPropagation,
    _resolve_nodata_handling,
    _validate_nodata_propagation,
)
from geoutils.operators.reducer import Mean, Reducer, _CallableReducer
from geoutils.operators.weighting import _with_error_structure
from geoutils.projtools import _affine_matmul, reproject_from_latlon
from geoutils.raster.referencing import _bbox, _coords, _res, _xy2ij

# Kriging needs a fitted model, so callers pass a Kriging object through the Interpolator choice
InterpolationMethod = RegularInterpolationMethod
InterpolationMethodLike = InterpolationMethod | Interpolator

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase
    from geoutils.raster.raster import Raster
    from geoutils.uncertainty.error_structure import ErrorStructure


def _destination_pixel_indices(
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int],
) -> tuple[NDArrayNum, NDArrayNum]:
    """
    Return source array indices at the centers of destination pixels.

    :param src_transform: Geotransform of the source array.
    :param dst_transform: Geotransform of the destination array.
    :param dst_shape: Height and width of the destination array.

    :return: Source row and column indices for every destination pixel center.
    """

    # Build destination pixel-center positions without retaining coordinate pairs as Python objects
    dst_cols, dst_rows = np.meshgrid(np.arange(dst_shape[1]) + 0.5, np.arange(dst_shape[0]) + 0.5)
    dst_x = dst_transform.a * dst_cols + dst_transform.b * dst_rows + dst_transform.c
    dst_y = dst_transform.d * dst_cols + dst_transform.e * dst_rows + dst_transform.f

    # Transform coordinates back to source pixels and place array index zero at the first center
    inverse = ~src_transform
    src_cols = inverse.a * dst_x + inverse.b * dst_y + inverse.c - 0.5
    src_rows = inverse.d * dst_x + inverse.e * dst_y + inverse.f - 0.5
    return src_rows, src_cols


def _interpolate_array(
    array: NDArrayNum,
    src_transform: rio.transform.Affine,
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int] | None = None,
    method: Literal["nearest", "linear", "bilinear"] = "linear",
    nodata_propagation: NodataPropagation = "nearest",
) -> NDArrayNum:
    """
    Interpolate an array onto another grid in the same coordinate reference system.

    This function separates coordinate mapping from value interpolation so it can be reused by same-CRS
    reprojection. The default reproduces GDAL nearest and bilinear nodata behavior, while ``ignore`` always uses
    available finite neighbors and ``propagate`` rejects outputs influenced by an invalid neighbor.

    :param array: Two- or three-dimensional source array, with bands on the first axis.
    :param src_transform: Geotransform of the source array.
    :param dst_transform: Geotransform of the destination array.
    :param dst_shape: Height and width of the destination array. Defaults to the source shape.
    :param method: Nearest-neighbor or linear interpolation. ``bilinear`` is an alias for ``linear``.
    :param nodata_propagation: Rule used to handle invalid source values.

    :return: Interpolated floating-point array.
    """

    # Normalize inputs before building the shared destination-to-source coordinate mapping
    source = np.asanyarray(array)
    if source.ndim not in (2, 3):
        raise ValueError("array must have two or three dimensions.")
    resolved_dst_shape = (source.shape[-2], source.shape[-1]) if dst_shape is None else dst_shape
    normalized_method: Literal["nearest", "linear"] = "linear" if method == "bilinear" else method
    if normalized_method not in ("nearest", "linear"):
        raise ValueError(f"Unknown interpolation method: {method!r}.")
    propagation = _validate_nodata_propagation(nodata_propagation)
    src_rows, src_cols = _destination_pixel_indices(src_transform, dst_transform, resolved_dst_shape)

    # Interpolate each band independently because nodata locations can differ between bands
    if source.ndim == 2:
        return _interpolate_array_band(source, src_rows, src_cols, normalized_method, propagation)
    bands = [
        _interpolate_array_band(source[band], src_rows, src_cols, normalized_method, propagation)
        for band in range(source.shape[0])
    ]
    return np.stack(bands)


# Dask as optional dependency
try:
    import dask
    import dask.array as da
    from dask import delayed
except ImportError:
    da = None

    def delayed(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake delayed decorator if dask is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


####################################################
# 1/ REGULAR GRID INTERPOLATION AT POINT COORDINATES
####################################################


# 1.1/ IN-MEMORY RASTER INTERPOLATION

# BASE FUNCTION FOR INTERP POINTS (WHOLE ARRAY IN MEMORY, USED BY CHUNKED FUNCTIONS + MAIN API)


def _point_index_type(x: NDArrayNum | Number, y: NDArrayNum | Number) -> type[np.float32] | type[np.float64]:
    """Use float64 pixel indices when the input coordinates need more than float32 precision."""

    coordinate_dtype = np.result_type(np.asarray(x), np.asarray(y), np.float32)
    return np.float64 if coordinate_dtype.itemsize > np.dtype(np.float32).itemsize else np.float32


@overload
def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    area_or_point: Literal["Area", "Point"] | None = None,
    method: InterpolationMethodLike | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "nearest",
    *,
    return_interpolator: Literal[False] = False,
    array_indices: tuple[NDArrayNum, NDArrayNum] | None = None,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    source_band: int = 1,
    _validity_only: bool = False,
    **kwargs: Any,
) -> NDArrayNum: ...


@overload
def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | None,
    area_or_point: Literal["Area", "Point"] | None = None,
    method: InterpolationMethodLike | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "nearest",
    *,
    return_interpolator: Literal[True],
    array_indices: tuple[NDArrayNum, NDArrayNum] | None = None,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    source_band: int = 1,
    _validity_only: bool = False,
    **kwargs: Any,
) -> Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


@overload
def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | None,
    area_or_point: Literal["Area", "Point"] | None = None,
    method: InterpolationMethodLike | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "nearest",
    *,
    return_interpolator: bool = False,
    array_indices: tuple[NDArrayNum, NDArrayNum] | None = None,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    source_band: int = 1,
    _validity_only: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]: ...


def _interp_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum] | None,
    area_or_point: Literal["Area", "Point"] | None = None,
    method: InterpolationMethodLike | None = None,
    dist_nodata_spread: Literal["half_order_up", "half_order_down"] | int | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    nodata_propagation: NodataPropagation = "nearest",
    return_interpolator: bool = False,
    array_indices: tuple[NDArrayNum, NDArrayNum] | None = None,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    source_band: int = 1,
    _validity_only: bool = False,
    **kwargs: Any,
) -> NDArrayNum | Callable[[tuple[NDArrayNum, NDArrayNum]], NDArrayNum]:
    """
    Interpolate a raster at point coordinates.

    This internal function can optionally reuse global pixel indices to work on chunks.
    The private validity-only option follows _resample_at_points(); its conversion happens within this loaded array.
    """

    # If interpolation method undefined, default to the global system config
    if method is None:
        method = config["interpolation_method"]
    method = _resolve_interpolator(method)

    # Convert availability within the current block, preserving masked cells without copying the source values
    if _validity_only:
        finite = np.isfinite(array)
        if np.ma.isMaskedArray(finite):
            finite = finite.filled(False)
        array = np.where(finite, np.float32(1), np.float32(np.nan))

    # If array is not a floating dtype (to support NaNs), convert dtype
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32)
    # If array is masked, fill with NaN without copy
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)

    # Check the public missing-data option before passing it to a built-in or custom method
    propagation = _validate_nodata_propagation(nodata_propagation)

    if isinstance(method, RasterConvolution):
        # GDAL-style kernels use source pixel spacing, including on a rotated raster grid
        def interpolate_convolution(x: NDArrayNum, y: NDArrayNum) -> NDArrayNum:
            """Sample a regular raster with the selected separable convolution kernel."""

            if array_indices is None:
                rows, columns = _xy2ij(
                    x,
                    y,
                    transform=transform,
                    area_or_point=area_or_point,
                    shift_area_or_point=shift_area_or_point,
                    op=np.float64,
                )
                rows, columns = rows - 0.5, columns - 0.5
            else:
                rows, columns = array_indices
            return method._interpolate_grid(array, rows, columns, propagation)

        if return_interpolator:

            def point_interpolator(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:
                """Accept array-axis coordinate order from the prepared interpolator API."""

                return interpolate_convolution(np.asarray(xi[1]), np.asarray(xi[0]))

            return point_interpolator

        if points is None:
            raise ValueError("Raster convolution requires target points.")
        return interpolate_convolution(np.asarray(points[0]), np.asarray(points[1]))

    # Built-in Interpolators use the existing SciPy array functions, which preserves their numerical behavior
    method_name = _regular_interpolation_method(method)

    # Moment interpolators evaluate uncertainty using the actual regular-grid coefficients in each loaded chunk
    if method_name is not None and getattr(method, "_requires_local_evaluation", False):
        if points is None or return_interpolator:
            raise ValueError("Local interpolation moments require target coordinates.")
        prepared, inputs, handling = _prepare_regular_interpolation_data(
            array,
            transform,
            points,
            method,
            area_or_point=area_or_point,
            shift_area_or_point=shift_area_or_point,
            nodata_propagation=propagation,
            array_indices=array_indices,
            band=source_band,
            source_index_offset=source_index_offset,
            source_shape=source_shape,
            index_type=_point_index_type(*points),
        )
        evaluator = method._with_regular_coefficients(prepared)
        result = evaluator.predict_batch(inputs, nodata_propagation=handling)
        # Apply the same geometric bounds and nodata mask as the ordinary numeric interpolation
        nominal = _interp_points_base(
            array,
            transform,
            points,
            area_or_point=area_or_point,
            method=method._wrapped_operator,
            shift_area_or_point=shift_area_or_point,
            nodata_propagation=propagation,
            dist_nodata_spread=dist_nodata_spread,
            array_indices=array_indices,
        )
        result[~np.isfinite(nominal)] = np.nan
        return result

    # Custom GridNeighbours methods run after the common data type and mask preparation above
    if method_name is None:
        if return_interpolator:
            raise ValueError("Interpolators using GridNeighbours cannot be returned as prepared SciPy interpolators.")
        if points is None:
            raise ValueError("Interpolators using GridNeighbours require target points.")
        return _resample_array_at_points(
            array=array,
            transform=transform,
            points=points,
            operator=method,
            area_or_point=area_or_point,
            shift_area_or_point=shift_area_or_point,
            nodata_propagation=propagation,
            dist_nodata_spread=dist_nodata_spread,
            band=source_band,
            source_index_offset=source_index_offset,
            source_shape=source_shape,
        )

    # Nearest and linear interpolation share the same finite-weight rules as same-grid interpolation
    if method_name in ("nearest", "linear"):
        selected_method = cast(Literal["nearest", "linear"], method_name)

        def interpolate_nearest_or_linear(x: NDArrayNum, y: NDArrayNum) -> NDArrayNum:
            """Interpolate point coordinates with the shared nearest or linear policy."""

            # Convert georeferenced coordinates to array indices before applying the common numeric kernel
            if array_indices is None:
                i, j = _xy2ij(
                    x,
                    y,
                    transform=transform,
                    area_or_point=area_or_point,
                    shift_area_or_point=shift_area_or_point,
                    op=_point_index_type(x, y),
                )
            else:
                i, j = array_indices
            return _interpolate_array_band(
                array=array,
                src_rows=i,
                src_cols=j,
                method=selected_method,
                nodata_propagation=propagation,
                dist_nodata_spread=dist_nodata_spread,
            )

        if return_interpolator:
            # Interpolators receive coordinates in array-axis order to match SciPy's existing interface
            def point_interpolator(xi: tuple[NDArrayNum, NDArrayNum]) -> NDArrayNum:
                return interpolate_nearest_or_linear(x=np.asarray(xi[1]), y=np.asarray(xi[0]))

            return point_interpolator

        assert points is not None
        return interpolate_nearest_or_linear(x=np.asarray(points[0]), y=np.asarray(points[1]))

    if not return_interpolator:
        assert points is not None
        x, y = points

    # Get lower-left corner coordinates
    xycoords = _coords(
        transform=transform,
        shape=(array.shape[0], array.shape[1]),
        area_or_point=area_or_point,
        grid=False,
        shift_area_or_point=shift_area_or_point,
    )

    # Let interpolation outside the bounds not raise any error by default
    if "bounds_error" not in kwargs.keys():
        kwargs.update({"bounds_error": False})
    # Return NaN outside image bounds
    if "fill_value" not in kwargs.keys():
        kwargs.update({"fill_value": np.nan})

    # Using direct coordinates, Y is the first axis, and we need to flip it
    scipy_interpolator = cast(ScipyInterpolator, method)._regular_grid_interpolator(
        points=(np.flip(xycoords[1], axis=0), xycoords[0]),
        values=array,
        dist_nodata_spread=dist_nodata_spread,
        nodata_propagation=propagation,
        bounds_error=kwargs["bounds_error"],
        fill_value=kwargs["fill_value"],
    )
    if return_interpolator:
        return scipy_interpolator
    return scipy_interpolator((y, x))  # type: ignore


def _resample_points_base(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[NDArrayNum, NDArrayNum] | None,
    *,
    method: Interpolator | Reducer,
    fractional_window: int | None = None,
    fractional_shape: Literal["square", "circular"] | None = None,
    **kwargs: Any,
) -> Any:
    """Apply an interpolator or reducer to one loaded raster band or worker tile."""

    if isinstance(method, Interpolator):
        return _interp_points_base(array, transform, points, method=method, **kwargs)
    if points is None:
        raise ValueError("Raster reduction requires target points.")

    # Same neighborhood and global cell IDs for eager arrays and worker tiles
    return _resample_array_at_points(
        array,
        transform,
        points,
        method,
        area_or_point=kwargs["area_or_point"],
        shift_area_or_point=False,
        nodata_propagation=kwargs["nodata_propagation"],
        dist_nodata_spread=kwargs["dist_nodata_spread"],
        neighborhood=cast(GridNeighbours, method.default_neighborhood),
        fractional_window=fractional_window,
        fractional_shape=fractional_shape,
        band=kwargs["source_band"],
        source_index_offset=kwargs.get("source_index_offset", (0, 0)),
        source_shape=kwargs.get("source_shape"),
    )


# 1.2/ DASK RASTER CHUNKS

# CHUNKED LOGIC: POINT INTERPOLATION ON REGULAR OR EQUAL GRID
# Notes at the date of April 2024:
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
    xres: float,
    yres: float,
    left: float,
    top: float,
) -> list[list[int]]:
    """Map blocks where each pair of interpolation coordinates will have to be computed."""

    # The argument "starts" contains the list of chunk first X/Y index for the full array, plus the last index
    ny, nx = num_chunks
    y_starts, x_starts = starts

    # We use one bucket per block, assuming a flattened blocks shape
    ind_per_block = [[] for _ in range(ny * nx)]
    for i, (x, y) in enumerate(zip(interp_x, interp_y)):
        # Use actual chunk boundaries because overlap can merge small edge chunks
        xb = int(np.searchsorted(x_starts, (x - left) / xres, side="right") - 1)
        yb = int(np.searchsorted(y_starts, (top - y) / yres, side="right") - 1)

        # Assign outer half pixels to the first block, matching the interpolation kernel's finite support
        if left - xres / 2 <= x < left:
            xb = 0
        if top < y <= top + yres / 2:
            yb = 0

        if 0 <= xb < nx and 0 <= yb < ny:
            ind_per_block[yb * nx + xb].append(i)

    return ind_per_block


@delayed
def _delayed_resample_points_block(
    arr_chunk: NDArrayNum,
    block_id: dict[str, Any],
    interp_coords: NDArrayNum,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Resample a loaded 2D block at its assigned point coordinates.
    """

    # Extract information out of block_id dictionary
    xs, ys, xres, yres = (block_id["xstart"], block_id["ystart"], block_id["xres"], block_id["yres"])

    # Reconstruct the transform from xi/yi/xres/yres
    transform = rio.transform.from_origin(xs, ys, xres, yres)

    # Interpolate to points by dispatching to base function
    interp_chunk = _resample_points_base(
        array=arr_chunk,
        transform=transform,
        points=(interp_coords[0, :], interp_coords[1, :]),
        **kwargs,
    )

    # And return the interpolated array
    return interp_chunk


def _dask_resample_points(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum],
    band: int,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Resample raster values at point coordinates using Dask chunks.

    This function harnesses the fact that a raster is defined on a regular (or equal) grid, and it is therefore
    faster than Xarray.interpn (especially for small sample sizes) and uses only a fraction of the memory usage.

    :param source_raster: Raster with Dask-backed data.
    :param band: Source band number, starting at one.
    :param points: Point(s) at which to interpolate raster value. If points fall outside of image, value
            returned is nan. Shape should be tuple of arrays.
    :param kwargs: Keyword arguments passed to _resample_points_base().

    :return: Array of raster value(s) interpolated at the given points.
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Empty selections stay lazy without reading source data
    output_dtype = (
        np.float64
        if isinstance(kwargs["method"], Reducer)
        else _interp_output_dtype(source_raster.dtype, validity_only=kwargs.get("_validity_only", False))
    )
    if len(points[0]) == 0:
        return da.from_array(np.empty(0, dtype=output_dtype), chunks="auto")
    darr = source_raster.data
    if darr.ndim != 2:
        darr = darr[band - 1]
    transform = source_raster.transform

    # Convert input to 2D array
    points_arr = np.vstack((points[0], points[1]))
    src_rows, src_cols = _xy2ij(
        points[0],
        points[1],
        transform=transform,
        area_or_point=kwargs["area_or_point"],
        shift_area_or_point=kwargs["shift_area_or_point"],
        op=_point_index_type(*points),
    )

    # Overlap for the neighborhood and nodata mask
    depth = max(_compute_resampling_overlap(kwargs["method"]), kwargs.get("dist_nodata_spread") or 0)
    res = _res(transform)
    bounds = _bbox(transform=transform, shape=darr.shape)
    left, top = bounds.left, bounds.top

    # Expand dask array for overlapping computations
    depths = (min(depth, darr.shape[0]), min(depth, darr.shape[1]))
    expanded = da.overlap.overlap(darr, depth={0: depths[0], 1: depths[1]}, boundary="nearest")

    # Recover core chunk boundaries after any automatic merging required for overlap
    core_chunks = [tuple(size - 2 * depths[index] for size in axis) for index, axis in enumerate(expanded.chunks)]
    starts = [cached_cumsum(axis, initial_zero=True) for axis in core_chunks]
    num_chunks = expanded.numblocks

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :],
        points_arr[1, :],
        starts,
        num_chunks,
        res[0],
        res[1],
        left,
        top,
    )

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks = expanded.to_delayed().ravel()

    # Build the block IDs by unravelling starting indexes for each block (Y is first axis)
    indexes_yi, indexes_xi = np.unravel_index(np.arange(len(blocks)), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [
        {
            "xstart": left + (starts[1][indexes_xi[i]] - depths[1]) * res[0],
            "ystart": top - (starts[0][indexes_yi[i]] - depths[0]) * res[1],
            "xres": res[0],
            "yres": res[1],
        }
        for i in range(len(blocks))
    ]

    # Compute values delayed
    used = [i for i in range(len(blocks)) if len(ind_per_block[i]) > 0]
    list_interp = []
    for i in used:
        # Translate global indices by integer offsets to keep interpolation weights identical across chunks
        block_kwargs = kwargs.copy()
        method_name = None if isinstance(kwargs["method"], Reducer) else _regular_interpolation_method(kwargs["method"])
        if method_name in ("nearest", "linear"):
            row_offset = starts[0][indexes_yi[i]] - depths[0]
            col_offset = starts[1][indexes_xi[i]] - depths[1]
            block_kwargs["array_indices"] = (
                src_rows[ind_per_block[i]] - row_offset,
                src_cols[ind_per_block[i]] - col_offset,
            )
        elif isinstance(kwargs["method"], (Interpolator, Reducer)):
            block_kwargs["source_index_offset"] = (
                starts[0][indexes_yi[i]] - depths[0],
                starts[1][indexes_xi[i]] - depths[1],
            )
            block_kwargs["source_shape"] = darr.shape
        list_interp.append(
            _delayed_resample_points_block(blocks[i], block_ids[i], points_arr[:, ind_per_block[i]], **block_kwargs)
        )

    # We concatenate and re-order in a delayed manner
    def _concat_reorder(list_vals, list_inds):  # type: ignore
        # Flatten outputs to 1D and concatenate
        vals = [np.asarray(v).ravel() for v in list_vals]
        vcat = np.concatenate(vals) if vals else np.array([], dtype=np.float32)

        # Build index array and argsort
        inds = (
            np.concatenate([np.asarray(ii, dtype=np.int64) for ii in list_inds])
            if list_inds
            else np.array([], dtype=np.int64)
        )
        order = np.argsort(inds)
        return vcat[order]

    # Get list of indexes only for used blocks
    list_inds_used = [ind_per_block[i] for i in used]
    joined = dask.delayed(_concat_reorder)(list_interp, list_inds_used)

    # Join into one array using a floating type whenever source values cannot represent NaN
    interp_points = da.from_delayed(joined, shape=(len(points[0]),), dtype=output_dtype)

    # Padded edge chunks repeat their outer cells, so restore the bounds of the complete source raster
    margin = 0 if isinstance(kwargs["method"], Reducer) else 0.5
    inside = (src_rows >= -margin) & (src_rows < darr.shape[0] - margin)
    inside &= (src_cols >= -margin) & (src_cols < darr.shape[1] - margin)
    interp_points = da.where(inside, interp_points, np.nan)

    return interp_points


# 1.3/ DASK POINT CLOUD TARGETS


def _empty_pointcloud_meta(data_name: str, crs: Any, dtype: DTypeLike) -> gpd.GeoDataFrame:
    """Build an empty GeoDataFrame for Dask point-cloud outputs."""

    # Dask uses this empty object to infer columns, geometry and data types
    return gpd.GeoDataFrame(
        data={data_name: pd.Series(dtype=dtype)},
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )


def _resampling_point_output(source_raster: RasterBase, x: NDArrayNum, y: NDArrayNum, values: Any) -> Any:
    """Build point output without computing Dask resampling values."""

    from geoutils.pointcloud import PointCloud

    if not hasattr(values, "compute"):
        return source_raster._cast_pointcloud_output(PointCloud.from_xyz(x, y, values, crs=source_raster.crs))

    # Known coordinates, delayed values
    dask_geopandas = import_optional("dask_geopandas", package_name="dask-geopandas")
    from geoutils.pointcloud.dataframe import _build_pointcloud_output, _import_dask_dataframe

    dask_dataframe = _import_dask_dataframe()
    meta = _empty_pointcloud_meta("z", source_raster.crs, values.dtype)
    partition = dask.delayed(gpd.GeoDataFrame)(
        data={"z": values}, geometry=gpd.points_from_xy(x, y), crs=source_raster.crs
    )
    dataframe = dask_dataframe.from_delayed([partition], meta=meta)
    output = dask_geopandas.from_dask_dataframe(dataframe, geometry="geometry")
    return _build_pointcloud_output(output, data_name="z", as_dataframe=True)


def _resample_points_partition(
    part: gpd.GeoDataFrame,
    source_raster: RasterBase,
    interp_options: dict[str, Any],
    extra_kwargs: dict[str, Any],
    data_name: str,
    out_crs: Any,
) -> gpd.GeoDataFrame:
    """Resample one point partition and return values with the same geometry and index."""

    # Preserve the planned output structure even when Dask sends an empty partition
    out_dtype = (
        np.float64
        if isinstance(interp_options["method"], Reducer)
        else _interp_output_dtype(source_raster.dtype, validity_only=extra_kwargs.get("_validity_only", False))
    )
    if len(part) == 0:
        return _empty_pointcloud_meta(data_name=data_name, crs=out_crs, dtype=out_dtype)

    # Convert partition geometries to the coordinate arrays used by raster interpolation
    x = np.atleast_1d(np.asarray(part.geometry.x.values))
    y = np.atleast_1d(np.asarray(part.geometry.y.values))
    i, j = _xy2ij(
        x,
        y,
        transform=source_raster.transform,
        area_or_point=source_raster.area_or_point,
        shift_area_or_point=interp_options["shift_area_or_point"],
        op=_point_index_type(x, y),
    )
    # Detect partitions with no raster overlap before constructing interpolation work
    # Include outer half pixels accepted by nearest and linear interpolation when selecting partitions
    operator = interp_options["method"]
    method_name = None if isinstance(operator, Reducer) else _regular_interpolation_method(operator)
    margin = 0.5 if method_name in {"nearest", "linear"} else 0
    ind_outofbounds: NDArrayBool = (i < -margin) | (j < -margin)
    ind_outofbounds |= (i >= source_raster.shape[0] - margin) | (j >= source_raster.shape[1] - margin)

    if np.count_nonzero(~ind_outofbounds) == 0:
        z = np.full(len(part), np.nan, dtype=out_dtype)
    else:
        # Reuse the regular interpolation path within the current point partition
        z = _resample_at_points(
            source_raster=source_raster,
            points=(x, y),
            as_array=True,
            **interp_options,
            **extra_kwargs,
        )
        # A Dask raster may return a lazy array that must finish inside this task
        if hasattr(z, "compute"):
            z = z.compute()

    # Retain the original geometry and index while adding interpolated values
    return gpd.GeoDataFrame(
        data={data_name: np.asarray(z)},
        geometry=part.geometry,
        crs=out_crs,
        index=part.index,
    )


def _resample_points_dask_pointcloud(
    source_raster: RasterBase,
    points: Any,
    method: InterpolationMethodLike | Reducer,
    band: int,
    input_latlon: bool,
    as_array: bool,
    nodata_handling: NodataChoice | None,
    shift_area_or_point: bool | None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None,
    return_interpolator: bool,
    extra_kwargs: dict[str, Any],
) -> Any:
    """Resample raster values at a Dask-GeoPandas point cloud."""

    # Reject options whose eager return type cannot be represented by partitions
    if return_interpolator:
        raise ValueError("Option 'return_interpolator' of interp_points cannot be used with Dask point-cloud inputs.")
    if input_latlon:
        raise ValueError("Argument 'input_latlon' is only supported for tuple point inputs.")

    # Import only after identifying a Dask input so the dependency remains optional
    import_optional("dask_geopandas", package_name="dask-geopandas")

    # Reproject lazily so every partition reaches the raster in the same CRS
    out_crs = source_raster.crs
    points_in_crs = points if points.crs == out_crs else points.to_crs(out_crs)
    data_name = "z"
    out_dtype = (
        np.float64
        if isinstance(method, Reducer)
        else _interp_output_dtype(source_raster.dtype, validity_only=extra_kwargs.get("_validity_only", False))
    )
    meta = _empty_pointcloud_meta(data_name=data_name, crs=out_crs, dtype=out_dtype)

    # Package stable interpolation options once for each partition task
    interp_options = {
        "method": method,
        "band": band,
        "input_latlon": False,
        "nodata_handling": nodata_handling,
        "shift_area_or_point": shift_area_or_point,
        "force_scipy_function": force_scipy_function,
        "return_interpolator": False,
    }
    # Map the eager partition helper while keeping the complete point cloud lazy
    out = points_in_crs.map_partitions(
        _resample_points_partition,
        source_raster,
        interp_options,
        extra_kwargs,
        data_name,
        out_crs,
        meta=meta,
    )

    if as_array:
        # Read lengths from input points so array sizing does not execute the interpolation tasks
        lengths = tuple(points.map_partitions(len).compute())
        values = out[data_name].to_dask_array(lengths=lengths)
        return da.ma.masked_invalid(values) if extra_kwargs.get("masked", False) else values

    # Import after package initialization because the point cloud package also imports interpolation
    from geoutils.pointcloud.dataframe import _build_pointcloud_output

    # Set point output metadata without computing the interpolation tasks
    return _build_pointcloud_output(out, data_name=data_name, as_dataframe=True)


###############################################
# 1.3.1/ EAGER OR DASK ARRAY POINT TARGETS
###############################################


def _resample_array_point_partition(source: Any, x: Any, y: Any, options: dict[str, Any]) -> Any:
    """Resample raster values at one raw point block without constructing geometry objects."""
    # A block outside the raster yields NaN without warning about the complete point input
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="All provided points were outside of raster bounds", category=UserWarning
        )
        values = _resample_at_points(source, (x, y), as_array=True, **options)
    # Nested raster work runs within this task so a single distributed worker cannot wait for itself
    return values.compute(scheduler="synchronous") if hasattr(values, "compute") else values


def _resample_array_points(source: Any, points: Any, as_array: bool, options: dict[str, Any]) -> Any:
    """Project array coordinates and resample their rows eagerly or in lazy point blocks."""
    if options.get("input_latlon"):
        raise ValueError("Argument 'input_latlon' is only supported for tuple point inputs.")
    if options.get("return_interpolator"):
        raise ValueError("Array point resampling requires explicit output values rather than an interpolator.")
    if options.get("mp_config") is not None:
        raise ValueError("Array point inputs use Dask chunks rather than multiprocessing resampling.")
    projected = points._dataset if points.crs == source.crs else points.reproject(crs=source.crs)
    x, y, _ = projected.pc.to_xyz()

    # Known point lengths allow block outputs without running their interpolation tasks
    if points._is_dask:
        dask = import_optional("dask")
        import dask.array as da

        chunked = next(
            value for value in (points._dataset, *points._dataset.coords.values()) if value.chunks is not None
        )
        lengths = chunked.chunks[0]
        axes = [da.asarray(axis).rechunk((lengths,)).to_delayed().ravel() for axis in (x, y)]
        dtype = (
            np.float64
            if isinstance(options["method"], Reducer)
            else _interp_output_dtype(source.dtype, validity_only=options.get("_validity_only", False))
        )
        blocks = [
            da.from_delayed(
                dask.delayed(_resample_array_point_partition)(source, x_block, y_block, options),
                shape=(length,),
                dtype=dtype,
            )
            for x_block, y_block, length in zip(*axes, lengths)
        ]
        values = da.concatenate(blocks)
    else:
        values = _resample_at_points(source, (x, y), as_array=True, **options)
    if as_array:
        return values
    return projected.pc.copy(new_array=values)


# 1.4/ MULTIPROCESSING RASTER CHUNKS

# SAME WITH MULTIPROCESSING


def _wrapper_multiproc_resample_per_block(
    rst: Raster,
    block_id: dict[str, Any],
    interp_coords: NDArrayNum,
    band: int = 1,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Read one raster tile and resample its selected band at the assigned points.

    Band selection and interpolation options follow _resample_at_points(); validity conversion stays inside its
    shared array kernel so workers never need a complete source validity raster.
    """

    # Extract information out of block_id dictionary
    tile_idx = block_id["tile_idx"]

    # Crop input raster for the given block
    rst_block = rst.icrop((tile_idx[2], tile_idx[0], tile_idx[3], tile_idx[1]))

    # Loaded rasters keep every band when cropped; unloaded rasters already read only the requested band
    array = rst_block.data
    if array.ndim == 3:
        array = array[band - 1]

    # Interpolate to points by dispatching to base function
    interp_chunk = _resample_points_base(
        array=array,
        transform=rst_block.transform,
        points=(interp_coords[0, :], interp_coords[1, :]),
        **kwargs,
    )

    # And return the interpolated array
    return interp_chunk


def _multiproc_resample_points(
    rst: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum],
    config: MultiprocConfig,
    band: int = 1,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Resample raster values at point coordinates using multiprocessing tiles.

    Band selection and interpolation options follow _resample_at_points(); config supplies tile sizes and the cluster.
    """

    # Empty selections need no worker tasks or source reads
    if len(points[0]) == 0:
        return np.empty(0, dtype=float)

    # Convert input to 2D array
    points_arr = np.vstack((points[0], points[1]))

    # Compute global indices only for methods that pass them directly to blocks
    src_indices = None
    method_name = None if isinstance(kwargs["method"], Reducer) else _regular_interpolation_method(kwargs["method"])
    if method_name in ("nearest", "linear"):
        src_indices = _xy2ij(
            points[0],
            points[1],
            transform=rst.transform,
            area_or_point=kwargs["area_or_point"],
            shift_area_or_point=kwargs["shift_area_or_point"],
            op=_point_index_type(*points),
        )

    # Overlap for the neighborhood and nodata mask
    depth = max(_compute_resampling_overlap(kwargs["method"]), kwargs.get("dist_nodata_spread") or 0)
    res = _res(rst.transform)
    bounds = _bbox(transform=rst.transform, shape=rst.shape)
    left, top = bounds.left, bounds.top

    # Get multiprocessing chunk sizes
    chunks = normalize_chunks(chunks=config.chunks, shape=rst.shape)

    # Get starting 2D index for each chunk of the full array
    # (mirroring what is done in block_id of dask.array.map_blocks)
    tiling = block_bounds_from_chunks(chunks=chunks, shape=rst.shape, overlap=depth)
    starts = [
        cached_cumsum(chunks[0], initial_zero=True),
        cached_cumsum(chunks[1], initial_zero=True),
    ]
    num_chunks = (tiling.shape[0], tiling.shape[1])
    num_blocks = np.prod(num_chunks)

    # Get samples indices per blocks
    ind_per_block = _get_interp_indices_per_block(
        points_arr[0, :],
        points_arr[1, :],
        starts,  # type: ignore
        num_chunks,
        res[0],
        res[1],
        left,
        top,
    )

    # Build the block IDs by unravelling starting indexes for each block
    indexes_xi, indexes_yi = np.unravel_index(np.arange(num_blocks), shape=(num_chunks[0], num_chunks[1]))
    block_ids = [{"tile_idx": tiling[indexes_xi[i], indexes_yi[i], :]} for i in range(num_blocks)]

    # Select the requested band for worker reads, restoring it after calculation or failure
    original_bands = rst.bands
    rst._bands = (band,)
    try:
        # Create tasks for multiprocessing
        tasks = []
        for i in range(len(block_ids)):
            # Reuse the full raster fractional indices instead of recalculating from each tile transform
            block_kwargs = kwargs.copy()
            if src_indices is not None:
                src_rows, src_cols = src_indices
                row_offset, _, col_offset, _ = block_ids[i]["tile_idx"]
                block_kwargs["array_indices"] = (
                    src_rows[ind_per_block[i]] - row_offset,
                    src_cols[ind_per_block[i]] - col_offset,
                )
            elif isinstance(kwargs["method"], (Interpolator, Reducer)):
                row_offset, _, col_offset, _ = block_ids[i]["tile_idx"]
                block_kwargs["source_index_offset"] = (row_offset, col_offset)
                block_kwargs["source_shape"] = rst.shape

            # Launch the task on the cluster to process each tile
            tasks.append(
                config.cluster.submit(
                    _wrapper_multiproc_resample_per_block,
                    rst,
                    block_ids[i],
                    points_arr[:, ind_per_block[i]],
                    band=band,
                    **block_kwargs,
                )
            )

        # Collect results
        try:
            list_interp = []
            # Iterate over the tasks and retrieve the processed results
            for results in tasks:
                interp = config.cluster.compute(results)
                list_interp.append(interp)
        except Exception as e:
            raise RuntimeError(f"Error retrieving interpolated segments from multiprocessing tasks: {e}")
    finally:
        rst._bands = original_bands

    # Concatenate outputs
    interp_points = np.concatenate(list_interp, axis=0)

    # Re-order per-block output points to match their original indices
    indices = np.concatenate(ind_per_block).astype(int)
    argsort = np.argsort(indices)
    interp_points = np.array(interp_points)[argsort]

    return interp_points


# 1.5/ INPUT CHECKS AND BACKEND DISPATCH

# MAIN API FUNCTION CHECKING USER INPUTS AND DISPATCHING TO BASE, DASK OR MULTIPROCESSING


def _prepare_resampling_options(
    source_raster: RasterBase,
    method: InterpolationMethodLike | Reducer | Callable[[NDArrayNum], float] | None,
    *,
    band: int,
    window: int | None,
    window_shape: Literal["square", "circular"] | None,
    coverage: GridCoverage | None,
    masked: bool,
    nodata_handling: NodataChoice | None,
    shift_area_or_point: bool | None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None,
    return_interpolator: bool,
    _validity_only: bool,
    error_structure: ErrorStructure | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Resolve the method, neighborhood and missing-data options shared by every resampling backend."""

    # Validate window options before loading any raster data
    if "neighborhood" in kwargs:
        raise TypeError("Set the neighborhood on the Interpolator or Reducer when constructing it.")
    if window_shape is not None and window_shape not in ("square", "circular"):
        raise ValueError("window_shape must be 'square' or 'circular'.")
    if window is not None and (isinstance(window, bool) or not float(window).is_integer()):
        raise ValueError("Window must be a whole number.")
    if window is not None and (window < 1 or window % 2 != 1):
        raise ValueError("Window must be an odd number.")
    if window is not None:
        window = int(window)

    # Standard means supply coefficients for fractional weights and observation errors
    operator: Interpolator | Reducer
    if isinstance(method, (Interpolator, Reducer)):
        operator = method
    elif callable(method):
        if coverage == "fractional" or error_structure is not None:
            if method not in (np.mean, np.nanmean, np.ma.mean):
                raise TypeError("Fractional area or error_structure requires a Reducer or a standard mean callable.")
            operator = Mean()
        else:
            operator = _CallableReducer(method, masked=masked)
    else:
        operator = _resolve_interpolator(config["interpolation_method"] if method is None else method)
    if error_structure is not None:
        operator = _with_error_structure(operator, error_structure)

    # Reducers use their window; interpolators use their natural stencil or configured neighbors
    fractional_window = None
    fractional_shape = None
    if isinstance(operator, Reducer):
        if kwargs:
            raise TypeError(f"Unknown reducer options: {', '.join(sorted(kwargs))}.")
        if return_interpolator:
            raise ValueError("A Reducer returns reduced values, not an interpolator.")
        if isinstance(operator.default_neighborhood, PointNeighbours):
            raise ValueError("PointNeighbours applies to point sources; use GridNeighbours for raster cells.")
        neighborhood = _configure_grid_neighbours(
            operator.default_neighborhood,
            size=window,
            shape=window_shape,
            coverage=coverage,
        )
        assert neighborhood is not None
        if neighborhood is not operator.default_neighborhood:
            operator = copy(operator)
            operator.default_neighborhood = neighborhood
        if neighborhood.coverage == "fractional" and masked:
            raise ValueError("Fractional area does not support masked output.")
        if neighborhood.coverage != "center":
            size = 2 * max(max(abs(row), abs(col)) for row, col in neighborhood.offsets) + 1
            offsets = set(neighborhood.offsets)
            fractional_shape = neighborhood.window_shape
            if fractional_shape is None:
                candidate_shapes: tuple[Literal["square", "circular"], ...] = ("square", "circular")
                for candidate_shape in candidate_shapes:
                    if offsets == set(GridNeighbours(size=size, shape=candidate_shape).offsets):
                        fractional_shape = candidate_shape
                        break
            if fractional_shape is None or offsets != set(GridNeighbours(size=size, shape=fractional_shape).offsets):
                raise ValueError("Area coverage requires a square or circular GridNeighbours window.")
            fractional_window = size
        shift_area_or_point = False
        order = None
        if nodata_handling is None:
            nodata_handling = "ignore"
        elif isinstance(nodata_handling, str) and nodata_handling.lower() == "nearest":
            raise ValueError("Nearest-source nodata handling requires an Interpolator.")
    else:
        if window is not None or window_shape is not None:
            raise ValueError("Window reduction requires a Reducer or callable.")
        if coverage is not None:
            raise ValueError("Grid coverage requires a Reducer.")
        regular_method = _regular_interpolation_method(operator)
        if regular_method is not None:
            _check_regular_grid_neighbours(operator, regular_method, warn=True)
        else:
            neighbours = _resolve_grid_neighbours_for_interpolator(operator, source_raster.transform)
            if operator.default_neighborhood is None:
                operator = copy(operator)
                operator.default_neighborhood = neighbours
        order = operator.interpolation_order
        if nodata_handling is None:
            nodata_handling = config["interpolation_nodata_handling"]
    propagation, spread_distance = _resolve_nodata_handling(nodata_handling=nodata_handling, order=order)

    # Common options for loaded arrays and worker tiles
    return {
        "area_or_point": source_raster.area_or_point,
        "method": operator,
        "dist_nodata_spread": spread_distance,
        "nodata_propagation": propagation,
        "shift_area_or_point": shift_area_or_point,
        "force_scipy_function": force_scipy_function,
        "return_interpolator": return_interpolator,
        "source_band": band,
        "_validity_only": _validity_only,
        "fractional_window": fractional_window,
        "fractional_shape": fractional_shape,
        **kwargs,
    }


def _prepare_resampling_points(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
    *,
    input_latlon: bool,
    boundless: bool,
    options: dict[str, Any],
) -> tuple[tuple[NDArrayNum, NDArrayNum] | None, NDArrayBool, bool]:
    """Normalize target coordinates and select points with enough source cells for resampling."""

    # Check and normalize input points
    pts_xy, input_scalar = _check_match_points(source_raster, points)

    # Extract raster metadata for later checks and conversions
    transform = source_raster.transform
    area_or_point = source_raster.area_or_point
    shape = source_raster.shape
    operator = options["method"]
    fractional_window = options["fractional_window"]
    shift_area_or_point = options["shift_area_or_point"]

    # Convert from latlon if necessary
    pts = pts_xy
    if input_latlon:
        pts = reproject_from_latlon(pts_xy, out_crs=source_raster.crs)

    # If we evaluate points (not returning interpolator), remove those outside of bounds
    # (Out of bounds points are hard to deal with for chunked operations otherwise)
    if not options["return_interpolator"]:
        if pts is None:
            raise ValueError("Input 'points' cannot be None if 'return_interpolator' is False.")
        x0, y0 = pts
        # Normalize to 1D arrays for typing + uniform downstream logic
        x: NDArrayNum = np.atleast_1d(np.asarray(x0))
        y: NDArrayNum = np.atleast_1d(np.asarray(y0))

        if isinstance(operator, Reducer):
            # Reducers use containing cells, without shifting the pixel interpretation
            j, i = _affine_matmul(~transform, (x, y))
        else:
            i, j = _xy2ij(
                x,
                y,
                transform=transform,
                area_or_point=area_or_point,
                shift_area_or_point=shift_area_or_point,
                op=_point_index_type(x, y),
            )

        # Retain the outer half pixels accepted by nearest and linear array interpolation
        method_name = None if isinstance(operator, Reducer) else _regular_interpolation_method(operator)
        margin = 0.5 if method_name in {"nearest", "linear"} else 0
        ind_outofbounds: NDArrayBool = (i < -margin) | (j < -margin)
        ind_outofbounds |= (i >= shape[0] - margin) | (j >= shape[1] - margin)

        # Optional full-window requirement, checked in the complete raster before tiling
        if isinstance(operator, Reducer) and not boundless:
            if fractional_window is not None:
                radius = fractional_window / 2
                ind_outofbounds |= (i - radius < 0) | (i + radius > shape[0])
                ind_outofbounds |= (j - radius < 0) | (j + radius > shape[1])
            else:
                neighborhood = cast(GridNeighbours, operator.default_neighborhood)
                window_offsets = np.asarray(neighborhood.offsets)
                lower, upper = window_offsets.min(axis=0), window_offsets.max(axis=0)
                ind_outofbounds |= (np.floor(i) + lower[0] < 0) | (np.floor(i) + upper[0] >= shape[0])
                ind_outofbounds |= (np.floor(j) + lower[1] < 0) | (np.floor(j) + upper[1] >= shape[1])

        # Warn before returning missing values for an entirely outside interpolation request
        if np.count_nonzero(~ind_outofbounds) == 0:
            if isinstance(operator, Interpolator):
                warnings.warn("All provided points were outside of raster bounds, returning only NaNs.")
        return (x, y), ~ind_outofbounds, input_scalar
    return None, np.empty(0, dtype=bool), input_scalar


def _resample_points_in_memory(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum] | None,
    band: int,
    **kwargs: Any,
) -> Any:
    """Load the selected band and calculate its point values or reusable interpolator."""

    # Empty selections need no source reads
    if points is not None and len(points[0]) == 0:
        return np.empty(0, dtype=float)

    # Load one band for the shared array kernel
    array = source_raster.data
    if array.ndim != 2:
        array = array[band - 1]
    return _resample_points_base(array, source_raster.transform, points, **kwargs)


def _format_resampling_output(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum] | None,
    inside: NDArrayBool,
    values: Any,
    *,
    operator: Interpolator | Reducer,
    input_scalar: bool,
    as_array: bool,
    masked: bool,
    validity_only: bool,
) -> Any:
    """Restore outside targets and construct scalar, array or point-cloud output without computing Dask values."""

    # Reusable interpolators have no target coordinates
    if points is None:
        return values
    x, y = points
    dtype = (
        np.float64
        if isinstance(operator, Reducer)
        else _interp_output_dtype(source_raster.dtype, validity_only=validity_only)
    )
    if not np.any(inside):
        dtype = np.float32 if validity_only else np.float64

    # Fill outside points after calculation, within a delayed task for Dask arrays
    def _restore_outside(values: NDArrayNum) -> NDArrayNum:
        """Insert calculated values at their original target positions."""

        output = np.full(len(x), np.nan, dtype=dtype)
        output[inside] = values
        return output

    if hasattr(values, "compute"):
        delayed_values = dask.delayed(_restore_outside)(values)
        output = da.from_delayed(delayed_values, shape=(len(x),), dtype=dtype)
    else:
        output = _restore_outside(values)

    # Preserve lazy values when constructing point clouds, scalars or masks
    if not as_array:
        return _resampling_point_output(source_raster, x, y, output)
    if input_scalar and isinstance(operator, Reducer):
        output = output[0]
    if masked:
        return da.ma.masked_invalid(output) if hasattr(output, "compute") else np.ma.masked_invalid(output)
    return output


def _resample_at_points(
    source_raster: RasterBase,
    points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
    method: InterpolationMethodLike | Reducer | Callable[[NDArrayNum], float] | None = None,
    band: int | None = 1,
    input_latlon: bool = False,
    as_array: bool = False,
    nodata_handling: NodataChoice | None = None,
    shift_area_or_point: bool | None = None,
    force_scipy_function: Literal["map_coordinates", "interpn"] | None = None,
    return_interpolator: bool = False,
    mp_config: MultiprocConfig | None = None,
    _validity_only: bool = False,
    coverage: GridCoverage | None = None,
    error_structure: ErrorStructure | None = None,
    boundless: bool = True,
    window: int | None = None,
    window_shape: Literal["square", "circular"] | None = None,
    masked: bool = False,
    **kwargs: Any,
) -> Any:
    """
    Check resampling inputs and dispatch to in-memory, Dask or MP calculation.

    _prepare_resampling_options() resolves the operator and its window; _prepare_resampling_points() normalizes
    coordinates and selects valid targets. _resample_points_in_memory(), _dask_resample_points() and
    _multiproc_resample_points() calculate values through the shared array kernel. _format_resampling_output()
    restores outside targets and builds the requested output, preserving lazy Dask results.

    :param _validity_only: Sample availability using float32 one for finite cells and NaN for missing cells.
        Conversion happens within each loaded band or worker tile, without loading a complete validity raster.
    """

    # Validate band selection before resolving method and neighborhood options
    if band is None:
        if coverage == "fractional" and source_raster.count != 1:
            raise ValueError("Select a band for fractional reduction of a multiband raster.")
        band = 1
    if isinstance(band, bool) or not isinstance(band, (int, np.integer)) or not 1 <= band <= source_raster.count:
        raise ValueError("band must select a source raster band, starting at one.")
    options = _prepare_resampling_options(
        source_raster,
        method,
        band=band,
        window=window,
        window_shape=window_shape,
        coverage=coverage,
        masked=masked,
        nodata_handling=nodata_handling,
        shift_area_or_point=shift_area_or_point,
        force_scipy_function=force_scipy_function,
        return_interpolator=return_interpolator,
        _validity_only=_validity_only,
        error_structure=error_structure,
        **kwargs,
    )

    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from interp_points(). To use Multiprocessing, open the file without chunks."
        )

    if (dask_backend or mp_backend) and return_interpolator:
        raise ValueError(
            "Option 'return_interpolator' of interp_points cannot be used with Dask or Multiprocessing, "
            "only with in-memory array."
        )

    point_interface = _get_pointcloud_interface(points)
    if getattr(point_interface, "_is_xr", False):
        return _resample_array_points(
            source_raster,
            point_interface,
            as_array,
            {
                "method": options["method"],
                "band": band,
                "input_latlon": input_latlon,
                "nodata_handling": nodata_handling,
                "shift_area_or_point": options["shift_area_or_point"],
                "force_scipy_function": force_scipy_function,
                "return_interpolator": return_interpolator,
                "coverage": coverage,
                "window": window,
                "window_shape": window_shape,
                "boundless": boundless,
                "masked": masked,
                "mp_config": mp_config,
                "_validity_only": _validity_only,
                **kwargs,
            },
        )

    # Dask point partitions prepare their own coordinates inside each task
    if is_dask_geodataframe(points):
        if mp_backend:
            raise ValueError("Dask point-cloud inputs cannot be combined with Multiprocessing interpolation.")
        return _resample_points_dask_pointcloud(
            source_raster=source_raster,
            points=points,
            method=options["method"],
            band=band,
            input_latlon=input_latlon,
            as_array=as_array,
            nodata_handling=nodata_handling,
            shift_area_or_point=options["shift_area_or_point"],
            force_scipy_function=force_scipy_function,
            return_interpolator=return_interpolator,
            extra_kwargs={
                **kwargs,
                "_validity_only": _validity_only,
                "coverage": coverage,
                "window": window,
                "window_shape": window_shape,
                "boundless": boundless,
                "masked": masked,
            },
        )

    # Normalize coordinates and select targets within the complete raster
    coordinates, inside, input_scalar = _prepare_resampling_points(
        source_raster, points, input_latlon=input_latlon, boundless=boundless, options=options
    )
    selected_points = None
    if coordinates is not None:
        selected_points = coordinates[0][inside], coordinates[1][inside]

    # Delegate source reads and calculation to the selected backend
    if mp_config is not None:
        assert selected_points is not None
        values = _multiproc_resample_points(source_raster, selected_points, mp_config, band=band, **options)
    elif dask_backend:
        assert selected_points is not None
        values = _dask_resample_points(source_raster, selected_points, band=band, **options)
    else:
        values = _resample_points_in_memory(source_raster, selected_points, band=band, **options)
    return _format_resampling_output(
        source_raster,
        coordinates,
        inside,
        values,
        operator=options["method"],
        input_scalar=input_scalar,
        as_array=as_array,
        masked=masked,
        validity_only=_validity_only,
    )
