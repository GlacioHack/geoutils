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

"""Functions for consistent input checks and dispatching."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyproj
import rasterio as rio

from geoutils._typing import NDArrayNum, Number
from geoutils.exceptions import (
    IgnoredGridWarning,
    InvalidBoundsError,
    InvalidCRSError,
    InvalidGridError,
    InvalidPointsError,
    InvalidResolutionError,
    InvalidShapeError,
)
from geoutils.projtools import (
    _get_bounds_projected,
    reproject_points,
)
from geoutils.raster.georeferencing import _cast_pixel_interpretation

if TYPE_CHECKING:
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.vector.vector import Vector, VectorLike

# Helpers for duck typing: Check if object has attribute (or through an accessor)
#################################################################################


def get_geo_attr(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """Retrieve an attribute from an object, or one of its accessors."""

    # Try direct attribute (Raster, Vector, PointCloud, or accessor class)
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)

    # Try accessors (rst, vct, pc)
    for accessor_name in accessors:
        accessor = getattr(obj, accessor_name, None)
        if accessor is not None and hasattr(accessor, attr_name):
            return getattr(accessor, attr_name)

    # Fallback
    raise AttributeError(
        f"Attribute '{attr_name}' not found on object {type(obj)} " f"or its potential accessors {accessors}."
    )


def has_geo_attr(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """Check if attribute exists for an object, or one of its accessors."""

    # Check direct attribute (Raster, Vector, PointCloud, or accessor class)
    if hasattr(obj, attr_name):
        return True

    # Check accessors (rst, vct, pc)
    for accessor_name in accessors:
        accessor = getattr(obj, accessor_name, None)
        if accessor is not None and hasattr(accessor, attr_name):
            return True

    return False


# Level 0 checks: directly on user input
########################################


def _check_crs(crs: Any) -> pyproj.CRS:
    """Function for checking input CRS consistently."""

    # Let Pyproj do the job with from user input
    try:
        crs = pyproj.CRS.from_user_input(crs)
    except pyproj.exceptions.CRSError as e:
        raise InvalidCRSError("Projection not recognized by Pyproj.") from e

    return crs


def _check_bounds(
    bbox: rio.coords.BoundingBox | tuple[Number, Number, Number, Number] | pd.DataFrame
) -> tuple[Number, Number, Number, Number]:
    """Helper function to check bounds value when provided as a sequence or bounding box object."""

    if isinstance(bbox, rio.coords.BoundingBox):
        xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top

    elif isinstance(bbox, pd.DataFrame):
        if all(c in bbox.columns for c in ["minx", "maxx", "miny", "maxy"]) and len(bbox) == 1:
            xmin, ymin, xmax, ymax = bbox["minx"][0], bbox["miny"][0], bbox["maxx"][0], bbox["maxy"][0]
        else:
            raise InvalidBoundsError(
                f"Bounding box as a dataframe must contain columns 'minx', 'maxx', 'miny', "
                f"'maxy' and be of length 1, got columns {bbox.columns} with length {len(bbox)}"
            )

    # If bbox is an iterable with 4 coordinates (excluding strings and bytes)
    elif isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)):
        if len(bbox) != 4:
            raise InvalidBoundsError(f"Bounding box must be a sequence of length 4, got a length of {len(bbox)}.")
        # Normalize input as tuple (to accept lists, arrays, etc)
        try:
            bounds = tuple(float(x) for x in bbox)
        except (TypeError, ValueError) as e:
            raise InvalidBoundsError("Bounding box sequence must be numeric.") from e

        xmin, ymin, xmax, ymax = bounds

        # Check input relative order
        if xmin >= xmax:
            raise InvalidBoundsError(
                f"Invalid bounding box xmin must be < xmax, got xmin={xmin}, xmax={xmax}. Did you pass the right "
                f"sequence order (xmin, ymin, xmax, ymax)?"
            )
        if ymin >= ymax:
            raise InvalidBoundsError(
                f"Invalid bounding box: ymin must be < ymax, got ymin={ymin}, ymax={ymax}. Did you pass the right "
                f"sequence order (xmin, ymin, xmax, ymax)?"
            )
    else:
        raise InvalidBoundsError(
            f"Cannot interpret bounding box input from object of type {type(bbox).__name__!r}. "
            "Expected a sequence (xmin, ymin, xmax, ymax) or a rasterio BoundingBox."
        )

    return xmin, ymin, xmax, ymax


def _check_resolution(res: Number | tuple[Number, Number]) -> tuple[Number, Number]:
    """Helper function on checking resolution input."""

    # Case 1: Single scalar resolution
    if isinstance(res, (float, int, np.floating, np.integer)):
        # Should be finite value
        if not np.isfinite(res):
            raise InvalidResolutionError(f"Resolution must be a finite number, got {res!r}.")
        # Should be strictly positive
        if res <= 0:
            raise InvalidResolutionError(f"Resolution must be strictly positive, got {res!r}.")
        return float(res), float(res)

    # Case 2: Sequence of two numbers for X and Y (xres, yres)
    if isinstance(res, Sequence):
        # Should be a sequence of two
        if len(res) != 2:
            raise InvalidResolutionError(
                f"Resolution must be a number or a sequence of two numbers, " f"got a sequence of length {len(res)}."
            )

        # Should be numeric values
        try:
            xres, yres = (float(r) for r in res)
        except (TypeError, ValueError) as e:
            raise InvalidResolutionError("Resolution values must be numeric.") from e

        # Should be finite values
        if not np.isfinite(xres) or not np.isfinite(yres):
            raise InvalidResolutionError("Resolution values must be finite numbers.")

        # Should be strictly positive
        if xres <= 0 or yres <= 0:
            raise InvalidResolutionError(
                f"Resolution values must be strictly positive, " f"got (xres={xres}, yres={yres})."
            )

        return float(xres), float(yres)

    # If none of the above
    raise InvalidResolutionError(
        f"Resolution must be a number or a sequence of two numbers, got object of type {type(res).__name__}"
    )


def _check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Helper function on checking resolution input."""

    # Shape should be a sequence of two integers (exclude strings and bytes sequences)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise InvalidShapeError(f"Shape must be a sequence of two integers, got {shape!r}.")
    if len(shape) != 2:
        raise InvalidShapeError(f"Shape must have length 2, got length {len(shape)}.")

    # Normalize any sequence into a tuple
    try:
        nrows, ncols = (int(x) for x in shape)
    except (TypeError, ValueError) as e:
        raise InvalidShapeError(f"Shape values must be integers, got {shape!r}.") from e

    # Should be positive
    if nrows < 0 or ncols < 0:
        raise InvalidShapeError(f"Shape values must be non-negative, got {shape!r}.")

    return nrows, ncols


def _as_1d_numeric_array(obj: Any, name: str) -> NDArrayNum:
    """Helper to check 1d numeric array-like objects."""

    # Normalize to numeric array if supported.
    try:
        arr = np.asarray(obj, dtype=float)
    except Exception as e:
        raise InvalidGridError(f"{name} coordinates must be numeric and array-like.") from e

    # Remove potential 1-size dimensions
    arr = arr.squeeze()

    # Check number of dimensions
    if arr.ndim > 1:
        raise InvalidGridError(f"{name} coordinates must be 1D, got shape {arr.shape}.")

    # Check size (at least 2 to define a grid)
    if arr.size < 2:
        raise InvalidGridError(f"{name} coordinates must contain at least 2 points, got size {arr.size}.")

    return arr


def _check_coords(coords: tuple[NDArrayNum, NDArrayNum]) -> tuple[tuple[NDArrayNum, NDArrayNum], tuple[Number, Number]]:
    """
    Helper function on checking and normalizing coordinates input.

    Also returns the resolution of the regular grid, to avoid duplicating the processing outside the function.
    """

    # Only accept sequences that can contain numbers (exclude strings and bytes)
    if not isinstance(coords, Sequence) or isinstance(coords, (str, bytes)):
        raise InvalidGridError(
            f"Grid coordinates must be a sequence of two array-like objects (x, y), got object of type "
            f"{type(coords).__name__}."
        )

    if len(coords) != 2:
        raise InvalidGridError(
            f"Grid coordinates must contain exactly two elements (x, y), got length of {len(coords)}."
        )

    # Normalize into arrays
    x = _as_1d_numeric_array(coords[0], "X")
    y = _as_1d_numeric_array(coords[1], "Y")

    # Regularity check
    dx = np.diff(x)
    dy = np.diff(y)

    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise InvalidGridError("Grid coordinates must be regular " "(equally spaced independently along x and y).")

    return (x, y), (dx[0], dy[0])


# Level 1 checks: Can accept a reference object to match or manual input
########################################################################


def _check_match_points(
    src: RasterBase | Vector,
    points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
) -> tuple[tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum], bool]:
    """Function for checking and normalizing input of match feature for points consistently.

    :param src: Source object (raster, vector, point cloud).
    :param points: Points object (tuple of arrays/numbers, or point cloud).

    :return: Tuple of point coordinates (X and Y), Boolean to know if inputs were scalar.
    """

    # If points implements "bounds" and "crs"
    if has_geo_attr(points, "geometry") and has_geo_attr(points, "crs"):
        crs = get_geo_attr(points, "crs")
        pts = reproject_points(
            (points.geometry.x.values, points.geometry.y.values), in_crs=crs, out_crs=src.crs  # type: ignore
        )
        input_scalar = False

        logging.debug(
            f"Match points input: using reference object of type {type(src).__name__!r} "
            f" that implements 'geometry' and 'crs'."
        )

    # Or if points is a sequence of arrays (excluding strings and bytes)
    elif isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        # Needs to be a sequence of length 2
        if not isinstance(points, Sequence) or len(points) != 2:
            raise InvalidPointsError(
                f"Expected a sequence of two array-like objects (x, y), " f"got object of type {type(points).__name__}."
            )

        # Get each member of the sequence of 2
        x, y = points
        # Check that they are numeric, 1-dimension, same-length array-like
        try:
            x_arr = np.asarray(x, dtype=float).squeeze()
            y_arr = np.asarray(y, dtype=float).squeeze()
            # If inputs are scalars
            if x_arr.ndim == 0 and y_arr.ndim == 0:
                x_arr = x_arr.reshape(1)
                y_arr = y_arr.reshape(1)
                input_scalar = True
            else:
                input_scalar = False

        except Exception as e:
            raise InvalidPointsError("Point coordinates must be numeric array-like objects.") from e

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise InvalidPointsError(
                "Point coordinates must be a sequence of two 1-dimensional array-like objects, "
                f"got dimensions of {x_arr.ndim} and {y_arr.ndim}."
            )

        if x_arr.shape[0] != y_arr.shape[0]:
            raise InvalidPointsError(
                f"Point coordinates must have the same length, got lengths of {x_arr.shape[0]}"
                f" and {y_arr.shape[0]}."
            )

        pts = x_arr, y_arr

        if input_scalar:
            logging.debug("Match points input: using provided tuple of scalars.")
        else:
            logging.debug("Match points input: using provided tuple of array-likes.")

    else:
        raise InvalidPointsError(
            f"Cannot interpret point input from object of type {type(points).__name__!r}. "
            "Expected a tuple of two 1-d arrays (X, Y coordinates), "
            "or a geospatial object implementing 'geometry' (only containing points) and 'crs' such as a point cloud."
        )

    return pts, input_scalar


def _check_match_bbox(
    src: RasterBase | Vector,
    bbox: RasterLike | VectorLike | rio.coords.BoundingBox | tuple[Number, Number, Number, Number],
) -> tuple[Number, Number, Number, Number]:
    """Function for checking and normalizing input of match feature on bounds consistently.

    :param src: Source object (raster, vector, point cloud) that has a .bounds attribute.
    :param bbox: Bounding box object (tuple or bounding box or raster, vector, point cloud).

    :return: Tuple of bounding box (xmin, ymin, xmax, ymax).
    """

    # If bbox implements "bounds" and "crs"
    if has_geo_attr(bbox, "bounds") and has_geo_attr(bbox, "crs"):
        bounds = _check_bounds(get_geo_attr(bbox, "bounds"))
        crs = _check_crs(get_geo_attr(bbox, "crs"))
        xmin, ymin, xmax, ymax = _get_bounds_projected(bounds=bounds, in_crs=crs, out_crs=src.crs)

        logging.debug(
            f"Match bbox input: using reference object of type {type(src).__name__!r} "
            f"that implements 'bounds' and 'crs'."
        )

        # If input has an area_or_point attribute, raise warning if inconsistent with source
        if has_geo_attr(bbox, "area_or_point") and hasattr(src, "area_or_point"):
            _cast_pixel_interpretation(src.area_or_point, get_geo_attr(bbox, "area_or_point"))

    # If bbox is a rasterio bounding box
    elif isinstance(bbox, (rio.coords.BoundingBox, Sequence)):
        xmin, ymin, xmax, ymax = _check_bounds(bbox)

        logging.debug("Match bbox input: using provided bounding box or sequence of 4.")

    # If none of the above, add full description in error
    else:
        raise InvalidBoundsError(
            f"Cannot interpret bounding box input from object of type {type(bbox).__name__!r}. "
            "Expected a sequence (xmin, ymin, xmax, ymax), a rasterio BoundingBox, "
            "or a geospatial object implementing 'bounds' and 'crs' such as a raster, vector or point cloud."
        )

    return xmin, ymin, xmax, ymax


def _grid_from_bounds_res(
    bounds: tuple[Number, Number, Number, Number], res: Number | tuple[Number, Number]
) -> tuple[tuple[int, int], rio.Affine]:
    """Helper function to check and get grid transform and shape from bounds and resolution."""

    # Check and normalize input bounds
    xmin, ymin, xmax, ymax = _check_bounds(bounds)
    # Check and normalize resolution
    res = _check_resolution(res)

    # Derive shape and transform
    shape = round((ymax - ymin) / res[1]), round((xmax - xmin) / res[0])
    transform = rio.transform.from_origin(xmin, ymax, res[0], res[1])
    return shape, transform


def _grid_from_bounds_shape(
    bounds: tuple[Number, Number, Number, Number], shape: tuple[int, int]
) -> tuple[tuple[int, int], rio.Affine]:
    """Helper function to get grid transform from bounds and shape."""

    xmin, ymin, xmax, ymax = _check_bounds(bounds)
    shape = _check_shape(shape)
    # Careful, shape inverted (width, height in Rasterio; height, width in NumPy)
    transform = rio.transform.from_bounds(xmin, ymin, xmax, ymax, shape[1], shape[0])
    return shape, transform


def _grid_from_src(
    dst_crs: pyproj.CRS,
    src: Any,
    shape: tuple[int, int] | None = None,
    res: tuple[Number, Number] | Number | None = None,
    bounds: tuple[Number, Number, Number, Number] | None = None,
) -> tuple[tuple[int, int], rio.Affine]:
    """
    Helper function to get default grid shape/transform from user inputs, including "fallback" from source.

    Note: This step is necessary ONLY IF:
      1. Output CRS differs from input CRS,
      2. We have incomplete target grid: we either don't know target resolution/shape, or don't know target
        bounds, or none of the two.
    """

    # Check and normalize inputs, and pass shape/res if they exist
    if shape is not None and res is not None:
        raise AssertionError(
            "Internal logic violated: Shape and res should not be defined at the same time. "
            "This is a bug, please report it."
        )
    if shape is not None:
        height, width = _check_shape(shape)  # Careful, height/width are inverted in Rasterio (width comes first)
    else:
        height, width = None, None
    if res is not None:
        res = _check_resolution(res)
    else:
        res = None
    if bounds is not None:
        bounds = _check_bounds(bounds)
    else:
        bounds = None

    # First, for a raster source, if all are the same, return exactly source transform and size
    # (to avoid approximation errors from calculations below)
    if hasattr(src, "transform") and isinstance(src.transform, rio.Affine):
        if (
            (dst_crs == src.crs)
            & ((shape is None) | (shape == src.shape))
            & ((res is None) | (res == src.res))
            & ((bounds is None) | (bounds == src.bounds))
        ):

            return src.shape, src.transform

    # If there is no input grid (i.e. no resampling involved), just build output grid directly from user inputs
    if not hasattr(src, "res"):
        if bounds is None:
            bounds = _get_bounds_projected(_check_bounds(src.bounds), in_crs=src.crs, out_crs=dst_crs)
        if res is not None:
            return _grid_from_bounds_res(bounds, res)
        elif shape is not None:
            return _grid_from_bounds_shape(bounds, shape)

    # If input source has no width/height, should exist from source object
    if not has_geo_attr(src, "width"):
        src_width = width
        src_height = height
    else:
        src_width = src.width
        src_height = src.height

    # If no output res/shape exists (only bounds passed, or no input at all), we keep the same output shape as input
    if res is None and shape is None:
        width, height = src.width, src.height

    # We let Rasterio figure out the default transform (i.e. resolution) in the new CRS with same size output
    transform, width, height = rio.warp.calculate_default_transform(
        src.crs,
        dst_crs,
        src_width,
        src_height,
        left=src.bounds.left,
        right=src.bounds.right,
        top=src.bounds.top,
        bottom=src.bounds.bottom,
        resolution=res,  # Only defined if shape is None
        dst_width=width,  # Only defined if res is None
        dst_height=height,  # Only defined if res is None
    )
    out_shape = height, width

    # If no target bounds were passed, this is already the desired output
    # (calculate_default_transform already accounted for res input, or width/height input, or none of the two)
    if bounds is None:
        return out_shape, transform
    # If target bounds exist, we use the transform above to deduce the actual resolution of output, and re-compute
    # the transform using our bounds
    else:
        # If only shape was defined, this is direct
        if shape is not None:
            return _grid_from_bounds_shape(bounds=bounds, shape=shape)
        # Otherwise, we need to calculate the new output shape given target window (bounds), rounded to nearest integer,
        # so that we match the exact bounds
        else:
            ref_win = rio.windows.from_bounds(*list(bounds), transform).round_lengths()
            out_shape = (int(ref_win.height), int(ref_win.width))
            # We force the resolution here
            if res is not None:
                _, transform = _grid_from_bounds_res(bounds=bounds, res=res)
                return out_shape, transform
            # We force the bounds here
            else:
                return _grid_from_bounds_shape(bounds=bounds, shape=out_shape)


def _grid_from_coords(coords: tuple[NDArrayNum, NDArrayNum]) -> tuple[tuple[int, int], rio.Affine]:
    """Helper function to get grid transform from coordinates."""

    # Check input
    coords, dxdy = _check_coords(coords)
    x, y = coords

    # Get affine transform
    shape = len(coords[0]), len(coords[1])
    return shape, rio.transform.from_origin(
        west=x.min(),
        north=y.max(),
        xsize=dxdy[0],
        ysize=dxdy[0],
    )


def _check_match_grid(
    src: RasterBase | Vector,
    ref: RasterLike | VectorLike | None,
    res: Number | tuple[Number, Number] | None,
    shape: tuple[int, int] | None,
    bounds: tuple[Number, Number, Number, Number] | None,
    coords: tuple[NDArrayNum, NDArrayNum] | None,
    crs: pyproj.CRS | None,
) -> tuple[tuple[int, int], rio.Affine, pyproj.CRS]:
    """
    Function for checking and normalizing input of match feature on grids consistently.

    Uses reference input as priority, then optional manual inputs ('res', 'shape', 'bounds', 'coords'),
    otherwise fallbacks to source object.

    Note that :
    - Both 'res' and 'shape' describe the resolution, and cannot be used together (mutually exclusive). Shape
        order corresponds to that of the NumPy array (height, width).
    - Grid coordinates 'coords' is equivalent to passing 'res'/'shape' + 'bounds', and should be used alone.
    - The unit to use for 'res' or 'coords' is that of the matching CRS: reference if passed, or 'crs' if passed,
        otherwise source object (own CRS).


    The grid can be defined the following ways:
    1. If a raster referenced is provided, match its grid (CRS, bounds and resolution/shape),
    2. If a vector or point cloud reference is provided, match its projected extent (bounds, CRS) and set the
        resolution/shape to that provided (or, if none provided and source is a raster, fallback to that of source),
    3. If only a resolution/shape is provided, match this resolution using source extent (bounds, CRS),
    4. If both resolution/shape and bounds are provided, matches this full grid definition, optionally in
        another CRS if provided (otherwise uses source).
    5. If grid coordinates are provided, matches this full grid definition, optionally in another CRS if provided
        (otherwise uses source).

    :param src: Source object (raster, vector, point cloud).
    :param ref: Reference object (raster, vector, point cloud) to match for defining the destination grid.
    :param res: Resolution of destination grid, in units of destination CRS (mutually exclusive with shape).
    :param shape: Shape of destination grid (mutually exclusive with bounds).
    :param bounds: Bounding box (xmin, ymin, xmax, ymax) of destination grid, in units of destination CRS.
    :param coords: Regular coordinates (X, Y) of destination grid (to provide alone) in units of destination CRS.
    :param crs: Coordinate reference system of destination grid.

    :return: Shape and transform.
    """

    # Case 1: If reference is passed
    ################################
    if ref is not None:

        if crs is not None:
            raise InvalidGridError("Either 'ref' or 'crs' must be provided, not both.")

        # If reference defines a complete grid (raster-like)
        # IMPORTANT! Geodataframe do implement "transform" (method), so we check if transform is an Affine object
        if (
            has_geo_attr(ref, "shape")
            and has_geo_attr(ref, "transform")
            and has_geo_attr(ref, "crs")
            and isinstance(get_geo_attr(ref, "transform"), rio.Affine)
        ):
            dst_shape = get_geo_attr(ref, "shape")
            dst_transform = get_geo_attr(ref, "transform")
            dst_crs = get_geo_attr(ref, "crs")

            logging.debug(
                f"Match grid input: using reference object of type {type(src).__name__!r} "
                f"that implements 'transform' and 'shape' and 'crs' (raster-like)."
            )

            # Check redundance of other arguments
            redundant = {
                "res": res is not None,
                "bounds": bounds is not None,
                "shape": shape is not None,
                "coords": coords is not None,
            }
            used = [name for name, flag in redundant.items() if flag]
            # Warn if any extra argument was used
            if used:
                msg = (
                    f"Reference input from object of type {type(ref).__name__!r} already "
                    f"defines a complete grid, ignoring inputs {', '.join(used)}. "
                    "Pass only 'ref' input to silence this warning."
                )
                warnings.warn(category=IgnoredGridWarning, message=msg)

            # If input has an area_or_point attribute, raise warning if inconsistent with source
            if has_geo_attr(ref, "area_or_point") and hasattr(src, "area_or_point"):
                _cast_pixel_interpretation(src.area_or_point, get_geo_attr(ref, "area_or_point"))

        # If reference only defines a partial grid (vector or point cloud-like)
        elif has_geo_attr(ref, "bounds") and has_geo_attr(ref, "crs"):
            dst_bounds = _check_bounds(get_geo_attr(ref, "bounds"))
            dst_crs = _check_crs(get_geo_attr(ref, "crs"))

            logging.debug(
                f"Match grid input: using reference object of type {type(src).__name__!r} "
                f"that implements only 'bounds' and 'crs' (vector-like)."
            )

            if res is not None and shape is not None:
                raise InvalidGridError(
                    f"Both 'res' and 'shape' were passed to define the grid resolution alongside object of type"
                    f" {type(ref).__name__!r} defining bounds and CRS. Only provide one of 'res' or 'shape'."
                )
            if res is None and shape is None:
                # If no resolution was defined but source has one (= it is a raster), fallback on source
                if not hasattr(src, "res"):
                    raise InvalidGridError(
                        f"Reference input from object of type {type(ref).__name__!r} only contains "
                        f"bounds and CRS, and thus requires a provided resolution 'res' or grid shape 'shape' to "
                        f"define a complete grid, but none was passed and source object of type"
                        f" {type(src).__name__!r} (fallback) has none."
                    )
                else:
                    logging.debug(
                        "Match grid input: no resolution defined alongside reference, fallback on "
                        "resolution of source object."
                    )
                    # We calculate for potential differing CRS and target bounds
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, bounds=dst_bounds)
            if coords is not None:
                msg = (
                    f"Reference input from object of type {type(ref).__name__!r} and provided 'res' or 'shape' already "
                    f"define a complete grid, ignoring inputs 'coords'. Pass only 'ref' and ('res' or 'shape') to "
                    f"silence this warning."
                )
                warnings.warn(category=IgnoredGridWarning, message=msg)

            # Resolution and shape: after the above, one or the other must not be None
            # (Note: Both are already defined in target CRS, so no need for projected calculations)
            if res is not None:
                dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, bounds=bounds, res=res)
            elif shape is not None:
                dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, bounds=bounds, shape=shape)

        else:
            raise InvalidGridError(
                f"Cannot interpret reference grid from object of type {type(ref).__name__!r}. The reference grid "
                f"should implement either 'transform', 'shape' and 'crs' (raster-like), or 'bounds' and "
                f"'crs' (vector-like) through its object or accessors. If not, provide these arguments separately."
            )

    # Case 2: No reference is passed, only manual arguments (fallbacks on source)
    #############################################################################
    else:

        # Get output CRS, fallback to source
        if crs is not None:
            dst_crs = _check_crs(crs)
        else:
            dst_crs = _check_crs(src.crs)

        # If (res or shape) and bounds are defined, from user or on source fallback
        if (res is not None or shape is not None or hasattr(src, "res")) and (
            bounds is not None or hasattr(src, "bounds")
        ):

            # If both res and shape passed, raise error
            if res is not None and shape is not None:
                raise InvalidGridError(
                    "Both output grid resolution 'res' and shape 'shape' were passed, while "
                    "both describe resolution, only define one or the other."
                )
            # If coordinates are also passed, raise error
            if coords is not None and ((res is not None or shape is not None) and bounds is not None):
                raise InvalidGridError(
                    "Both 'coords' and ('res' or 'shape' + 'bounds) arguments were passed, while "
                    "both define a complete grid, only define one or the other."
                )

            # If coords exists, other arguments were insufficient to define a full grid (or would have failed above)
            # So we trigger fallback, but coords takes priority over fallback, so we skip if it exists
            if coords is None:

                # If user-input was passed
                if res is not None and bounds is not None:
                    logging.debug("Match grid input: using bounds and resolution to derive grid.")
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, bounds=bounds, res=res)
                if shape is not None and bounds is not None:
                    logging.debug("Match grid input: using bounds and shape to derive grid.")
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, bounds=bounds, shape=shape)

                # Fallback to source if res/shape or bounds undefined
                if bounds is None and (shape is not None or res is not None):
                    logging.debug("Match grid input: no bounds defined, fallback on source object.")
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, shape=shape, res=res)
                if bounds is not None and (res is None and shape is None):
                    logging.debug("Match grid input: no resolution defined, fallback on source object.")
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, bounds=bounds)
                if bounds is None and (res is None and shape is None):
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src)

        # If coordinates are defined
        if coords is not None:

            # Get redundant arguments (that could never define a full grid based on checks above)
            redundant = {
                "res": res is not None,
                "bounds": bounds is not None,
                "shape": shape is not None,
            }
            used = [name for name, flag in redundant.items() if flag]
            if used:
                msg = (
                    f"Grid coordinates 'coords' already defines a complete grid, ignoring inputs {', '.join(used)}. "
                    "Pass only 'coords' input to silence this warning."
                )
                warnings.warn(category=IgnoredGridWarning, message=msg)

            logging.debug("Match grid input: using coordinates to derive grid.")
            # Coordinates should be in units of output CRS, no ambiguity bounds/res, can compute directly from
            # coordinates
            dst_shape, dst_transform = _grid_from_coords(coords)

        if coords is None and not (
            (res is not None or shape is not None or hasattr(src, "res"))
            and (bounds is not None or hasattr(src, "bounds"))
        ):
            raise InvalidGridError(
                "Insufficient inputs to define a complete grid, which requires either: 1/ A raster-like object as "
                "reference, or 2/ A vector-like object as reference along with a provided "
                "resolution or shape, or 3/ A provided resolution or shape (use bounds from source object),"
                " 4/ A provided resolution or shape and provided bounds, "
                "or 5/ Provided grid coordinates; the last two optionally with a CRS (if different than source)."
            )

    return dst_shape, dst_transform, dst_crs
