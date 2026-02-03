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

from collections.abc import Sequence
import warnings
from typing import Any, TYPE_CHECKING
from numbers import Number

import numpy as np
import pandas as pd
import pyproj
import rasterio as rio

from geoutils.projtools import _get_bounds_projected, reproject_from_latlon, reproject_points
from geoutils.raster.georeferencing import _cast_pixel_interpretation
from geoutils._typing import NDArrayNum

from geoutils.exceptions import (InvalidGridError, InvalidCRSError, InvalidBoundsError, InvalidPointsError,
                                 InvalidResolutionError, InvalidShapeError, IgnoredGridWarning)

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike, RasterBase
    from geoutils.vector.vector import VectorLike, Vector
    from geoutils.pointcloud.pointcloud import PointCloudLike

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

# Level 0 checks: direcly on user input
#######################################

def _check_crs(crs: Any) -> pyproj.CRS:
    """Function for checking input CRS consistently."""

    # Let Pyproj do the job with from user input
    try:
        crs = pyproj.CRS.from_user_input(crs)
    except (pyproj.exceptions.CRSError) as e:
        raise InvalidCRSError("Projection not recognized by Pyproj.") from e

    return crs

def _check_bounds(bbox: rio.coords.BoundingBox | tuple[Number, Number, Number, Number] | pd.DataFrame) \
        -> tuple[Number, Number, Number, Number]:
    """Helper function to check bounds value when provided as a sequence or bounding box object."""

    if isinstance(bbox, rio.coords.BoundingBox):
        xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top

    elif isinstance(bbox, pd.DataFrame):
        if all(c in bbox.columns for c in ["minx", "maxx", "miny", "maxy"]) and len(bbox) == 1:
            xmin, ymin, xmax, ymax = bbox["minx"][0], bbox["miny"][0], bbox["maxx"][0], bbox["maxy"][0]
        else:
            raise InvalidBoundsError(
                f"Bounding box as a dataframe must contain columns 'minx', 'maxx', 'miny', "
                f"'maxy' and be of length 1, got columns {bbox.columns} with length {len(bbox)}")

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
    if isinstance(res, Number):
        # Should be finite value
        if not np.isfinite(res):
            raise InvalidResolutionError(
                f"Resolution must be a finite number, got {res!r}."
            )
        # Should be strictly positive
        if res <= 0:
            raise InvalidResolutionError(
                f"Resolution must be strictly positive, got {res!r}."
            )
        return res, res

    # Case 2: Sequence of two numbers for X and Y (xres, yres)
    if isinstance(res, Sequence):
        # Should be a sequence of two
        if len(res) != 2:
            raise InvalidResolutionError(
                f"Resolution must be a number or a sequence of two numbers, "
                f"got a sequence of length {len(res)}."
            )

        # Should be numeric values
        try:
            xres, yres = (float(r) for r in res)
        except (TypeError, ValueError) as e:
            raise InvalidResolutionError(
                "Resolution values must be numeric."
            ) from e

        # Should be finite values
        if not np.isfinite(xres) or not np.isfinite(yres):
            raise InvalidResolutionError(
                "Resolution values must be finite numbers."
            )

        # Should be strictly positive
        if xres <= 0 or yres <= 0:
            raise InvalidResolutionError(
                f"Resolution values must be strictly positive, "
                f"got (xres={xres}, yres={yres})."
            )

        return xres, yres

    # If none of the above
    raise InvalidResolutionError(
        f"Resolution must be a number or a sequence of two numbers, got object of type {type(res).__name__}"
    )

def _check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """Helper function on checking resolution input."""

    # Shape should be a sequence of two integers (exclude strings and bytes sequences)
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise InvalidShapeError(
            f"Shape must be a sequence of two integers, got {shape!r}."
        )
    if len(shape) != 2:
        raise InvalidShapeError(
            f"Shape must have length 2, got length {len(shape)}."
        )

    # Normalize any sequence into a tuple
    try:
        nrows, ncols = (int(x) for x in shape)
    except (TypeError, ValueError) as e:
        raise InvalidShapeError(
            f"Shape values must be integers, got {shape!r}."
        ) from e

    # Should be positive
    if nrows < 0 or ncols < 0:
        raise InvalidShapeError(
            f"Shape values must be non-negative, got {shape!r}."
        )

    return nrows, ncols

def _as_1d_numeric_array(obj: Any, name: str) -> NDArrayNum:
    """Helper to check 1d numeric array-like objects."""

    # Normalize to numeric array if supported.
    try:
        arr = np.asarray(obj, dtype=float)
    except Exception as e:
        raise InvalidGridError(
            f"{name} coordinates must be numeric and array-like."
        ) from e

    # Remove potential 1-size dimensions
    arr = arr.squeeze()

    # Check number of dimensions
    if arr.ndim > 1:
        raise InvalidGridError(
            f"{name} coordinates must be 1D, got shape {arr.shape}."
        )

    # Check size (at least 2 to define a grid)
    if arr.size < 2:
        raise InvalidGridError(
            f"{name} coordinates must contain at least 2 points, got size {arr.size}."
        )

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
        raise InvalidGridError(
            "Grid coordinates must be regular "
            "(equally spaced independently along x and y)."
        )

    return (x, y), (dx[0], dy[0])

# Level 1 checks: Can accept a reference object to match or manual input
########################################################################

def _check_match_points(source_obj: RasterBase | Vector,
                        points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
                        ) -> tuple[tuple[NDArrayNum, NDArrayNum], bool]:
    """Function for checking match feature for points consistently.

    :param source_obj: Source object (raster, vector, point cloud).
    :param points: Points object (tuple of arrays/numbers, or point cloud).

    :return: Tuple of point coordinates (X and Y), Boolean to know if inputs were scalar.
    """

    # If points implements "bounds" and "crs"
    if has_geo_attr(points, "geometry") and has_geo_attr(points, "crs"):
        crs = get_geo_attr(points, "crs")
        pts = reproject_points((points.geometry.x.values, points.geometry.y.values), in_crs=crs,
                               out_crs=source_obj.crs)
        input_scalar = False

    # Or if points is a sequence of arrays (excluding strings and bytes)
    elif isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        # Needs to be a sequence of length 2
        if not isinstance(points, Sequence) or len(points) != 2:
            raise InvalidPointsError(f"Expected a sequence of two array-like objects (x, y), "
                                     f"got object of type {type(points).__name__}.")

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
            raise InvalidPointsError("Point coordinates must be a sequence of two 1-dimensional array-like objects, "
                                     f"got dimensions of {x_arr.ndim} and {y_arr.ndim}.")

        if x_arr.shape[0] != y_arr.shape[0]:
            raise InvalidPointsError(f"Point coordinates must have the same length, got lengths of {x_arr.shape[0]}"
                                     f" and {y_arr.shape[0]}.")

        pts = x_arr, y_arr

    else:
        raise InvalidPointsError(
            f"Cannot interpret point input from object of type {type(points).__name__!r}. "
            "Expected a tuple of two 1-d arrays (X, Y coordinates), "
            "or a geospatial object implementing 'geometry' (only containing points) and 'crs' such as a point cloud."
        )

    return pts, input_scalar

def _check_match_bbox(source_obj: RasterBase | Vector,
                      bbox: RasterLike | VectorLike | rio.coords.BoundingBox | tuple[Number, Number, Number,
                        Number]) \
        -> tuple[Number, Number, Number, Number]:
    """Function for checking match feature on bounds consistently.

    :param source_obj: Source object (raster, vector, point cloud) that has a .bounds attribute.
    :param bbox: Bounding box object (tuple or bounding box or raster, vector, point cloud).

    :return: Tuple of bounding box (xmin, ymin, xmax, ymax).
    """

    # If bbox implements "bounds" and "crs"
    if has_geo_attr(bbox, "bounds") and has_geo_attr(bbox, "crs"):
        bounds = _check_bounds(get_geo_attr(bbox, "bounds"))
        crs = _check_crs(get_geo_attr(bbox, "crs"))
        xmin, ymin, xmax, ymax = _get_bounds_projected(bounds=bounds, in_crs=crs, out_crs=source_obj.crs)

        # If input has an area_or_point attribute, raise warning if inconsistent with source
        if has_geo_attr(bbox, "area_or_point") and hasattr(source_obj, "area_or_point"):
            _cast_pixel_interpretation(source_obj.area_or_point, get_geo_attr(bbox, "area_or_point"))

    # If bbox is a rasterio bounding box
    elif isinstance(bbox, (rio.coords.BoundingBox, Sequence)):
        xmin, ymin, xmax, ymax = _check_bounds(bbox)

    # If none of the above, add full description in error
    else:
        raise InvalidBoundsError(
            f"Cannot interpret bounding box input from object of type {type(bbox).__name__!r}. "
            "Expected a sequence (xmin, ymin, xmax, ymax), a rasterio BoundingBox, "
            "or a geospatial object implementing 'bounds' and 'crs' such as a raster, vector or point cloud."
        )

    return xmin, ymin, xmax, ymax


def _grid_from_bounds_res(bounds: tuple[Number, Number, Number, Number], res: Number | tuple[Number, Number]) \
        -> tuple[tuple[int, int], rio.Affine]:
    """Helper function to check and get grid transform and shape from bounds and resolution."""

    # Check and normalize input bounds
    xmin, ymin, xmax, ymax = _check_bounds(bounds)
    # Check and normalize resolution
    res = _check_resolution(res)

    # Derive shape and transform
    shape = round((xmax - xmin) / res[0]), round((ymax - ymin) / res[1])
    return shape, rio.transform.from_bounds(xmin, ymin, xmax, ymax, shape[0], shape[1])

def _grid_from_bounds_shape(bounds: tuple[Number, Number, Number, Number], shape: tuple[int, int]) -> rio.Affine:
    """Helper function to get grid transform from bounds and shape."""

    xmin, ymin, xmax, ymax = _check_bounds(bounds)
    return rio.transform.from_bounds(xmin, ymin, xmax, ymax, shape[0], shape[1])

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

def _check_match_grid(source_obj: RasterBase | Vector,
                      ref: RasterLike | VectorLike | None,
                      res: Number | tuple[Number, Number] | None,
                      shape: tuple[int, int] | None,
                      bounds: tuple[Number, Number, Number, Number] | None,
                      coords: tuple[NDArrayNum, NDArrayNum] | None,
                      crs: pyproj.CRS | None) -> tuple[tuple[int, int], rio.Affine, pyproj.CRS]:
    """Function for checking match feature on grids consistently.

    :param source_obj: Source object (raster, vector, point cloud).
    :param ref: Reference object (raster, vector, point cloud) to match for defining the grid.
    :param res: Resolution of grid (to provide with bounds; only needed if shape not provided).
    :param shape: Shape of grid (to provide with bounds; only needed if resolution not provided).
    :param bounds: Bounding box (xmin, ymin, xmax, ymax) of grid.
    :param coords: Regular coordinates (X, Y) of grid (to provide alone, nothing else required).
    :param crs: Coordinate reference system of grid.

    The grid can be defined with the following options:
    - Match a raster reference (CRS, bounds and resolution/shape),
    - Match a vector or point cloud reference extent (bounds, CRS) along with a provided resolution/shape,
    - Match provided bounds and resolution/shape, optionally defined in another CRS.
    - Match provided grid coordinates, optionally defined in another CRS.

    :return: Shape and transform.
    """

    # Case 1: If reference is passed
    ################################
    if ref is not None:

        # If reference defines a complete grid (raster-like)
        # IMPORTANT! Geodataframe do implement "transform" (method), so we check if transform is an Affine object
        if has_geo_attr(ref, "shape") and has_geo_attr(ref, "transform") and has_geo_attr(ref, "crs")\
                and isinstance(get_geo_attr(ref, "transform"), rio.Affine):
            ref_shape = get_geo_attr(ref, "shape")
            ref_transform = get_geo_attr(ref, "transform")
            ref_crs = get_geo_attr(ref, "crs")

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
                    "Pass only 'ref' input to silence this warning.")
                warnings.warn(category=IgnoredGridWarning, message=msg)

        # If reference only defines a partial grid (vector or point cloud-like)
        elif has_geo_attr(ref, "bounds") and has_geo_attr(ref, "crs"):
            ref_bounds = get_geo_attr(ref, "bounds")
            ref_crs = get_geo_attr(ref, "crs")
            if res is None and shape is None:
                raise InvalidGridError(
                    f"Reference input from object of type {type(ref).__name__!r} only contains "
                    f"bounds and CRS, and thus requires a provided resolution 'res' or grid shape 'shape' to define a "
                    f"complete grid, but none was passed.")
            elif res is not None and shape is not None:
                raise InvalidGridError(
                    f"Both 'res' and 'shape' were passed to define the grid resolution alongside object of type"
                    f" {type(ref).__name__!r} defining bounds and CRS. Only provide one of 'res' or 'shape'.")
            if coords is not None:
                msg = (
                    f"Reference input from object of type {type(ref).__name__!r} alongside 'res' or 'shape' already "
                    f"defines a complete grid, ignoring inputs 'coords'. Pass only 'ref' and ('res' or 'shape') to "
                    f"silence this warning.")
                warnings.warn(category=IgnoredGridWarning, message=msg)

            # Resolution and shape: after the above, one or the other must not be None
            if res is not None:
                ref_shape, ref_transform = _grid_from_bounds_res(ref_bounds, res)
            else:
                ref_shape = _check_shape(shape)
                ref_transform = _grid_from_bounds_shape(ref_bounds, shape)

        else:
            raise InvalidGridError(
                f"Cannot interpret reference grid from object of type {type(ref).__name__!r}. The reference grid "
                f"should implement either 'transform', 'shape' and 'crs' (raster-like), or 'bounds' and "
                f"'crs' (vector-like) through its object or accessors. If not, provide these arguments separately.")

    # Case 2: No reference is passed, only manual arguments
    #######################################################
    else:
        # If CRS is not defined
        if crs is not None:
            ref_crs = _check_crs(crs)
        else:
            ref_crs = source_obj.crs

        # If (res or shape) and bounds are defined
        if (res is not None or shape is not None) and bounds is not None:

            # Ensure only one of the two is defined
            if res is not None and shape is not None:
                raise InvalidGridError("Both output grid resolution 'res' and shape 'shape' were passed, while "
                                       "both describe resolution, only define one or the other.")
            # If coordinates are passed, raise warning that they are ignored
            if coords is not None:
                raise InvalidGridError("Both 'coords' and ('res' or 'shape' + 'bounds) arguments were passed, while "
                                       "both define a complete grid, only define one or the other.")

            if res is not None:
                ref_shape, ref_transform = _grid_from_bounds_res(bounds, res)
            else:
                ref_shape = _check_shape(shape)
                ref_transform = _grid_from_bounds_shape(bounds, shape)

        elif coords is not None:

            # Get redundant arguments (should never define a full grid based on above)
            redundant = {
                "res": res is not None,
                "bounds": bounds is not None,
                "shape": shape is not None,
            }
            used = [name for name, flag in redundant.items() if flag]
            if used:
                msg = (
                    f"Grid coordinates 'coords' already defines a complete grid, ignoring inputs {', '.join(used)}. "
                    "Pass only 'coords' input to silence this warning.")
                warnings.warn(category=IgnoredGridWarning, message=msg)

            ref_shape, ref_transform = _grid_from_coords(coords)

        else:
            raise InvalidGridError(
                "Insufficient inputs to define a complete grid, which requires either: 1/ A raster-like object as "
                "reference, or 2/ A vector-like object as reference along with a provided "
                "resolution or shape, or 3/ Bounds along with resolution or shape, "
                "or 4/ Grid coordinates; the last two optionally with a CRS (if different than source).")

    return ref_shape, ref_transform, ref_crs