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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast, get_args

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import xarray as xr
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from geoutils._typing import DTypeLike, NDArrayNum, Number, ResamplingMethod
from geoutils.exceptions import (
    CRSMismatchError,
    GeoUtilsInputError,
    GridMismatchError,
    IgnoredGridWarning,
    InvalidBoundsError,
    InvalidCRSError,
    InvalidGridError,
    InvalidNodataError,
    InvalidPointsError,
    InvalidResamplingError,
    InvalidResolutionError,
    InvalidShapeError,
    NoValidDataError,
    PointsMismatchError,
    ShapeMismatchError,
    _warn,
)
from geoutils.projtools import _get_bounds_projected, reproject_points
from geoutils.raster.referencing import _cast_pixel_interpretation, _check_affine_georeferencing

if TYPE_CHECKING:
    from geoutils.operators.nodata import NodataPropagation
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.vector.vector import Vector, VectorLike

# Helpers for recognizing optional Dask objects without importing Dask
################################################


def is_dask_dataframe(obj: Any) -> bool:
    """Return True for Dask DataFrame-like objects without importing optional dependencies."""

    # Dask and Dask-GeoPandas expose stable public module prefixes
    return obj.__class__.__module__.startswith(("dask.dataframe", "dask_geopandas"))


def is_dask_array(obj: Any) -> bool:
    """Return True for Dask Array-like objects without importing optional dependencies."""

    # Inspecting the module keeps the check usable when Dask is not installed
    return obj.__class__.__module__.startswith("dask.array")


def is_dask_geodataframe(obj: Any) -> bool:
    """Return True for Dask-GeoPandas GeoDataFrame-like objects without importing Dask-GeoPandas."""

    # Prefer the specific package marker when it is available
    if obj.__class__.__module__.startswith("dask_geopandas"):
        return True

    # Retain compatibility with dataframe wrappers exposing geospatial methods
    return is_dask_dataframe(obj) and hasattr(obj, "geometry") and hasattr(obj, "to_crs")


# Helpers for duck typing: Check if object has attribute (or through an accessor)
#################################################################################


def _get_registered_accessor(obj: Any, name: str) -> Any:
    """Choose point or raster accessors from Xarray dimensions before attempting their initialization."""
    if isinstance(obj, xr.DataArray):
        from geoutils.pointcloud.referencing import _point_coordinates

        try:
            _point_coordinates(obj)
            point_layout = obj.ndim == 1
        except (AttributeError, ValueError):
            point_layout = False
        if (name == "pc" and not point_layout) or (name == "rst" and point_layout):
            return None
    accessor = getattr(obj, name, None)
    if isinstance(obj, xr.Dataset) and name in ("rst", "pc") and accessor is not None:
        if not accessor.compatible_variables:
            return None
    return accessor


def get_geo_interface(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """
    Return the object or accessor providing a requested geospatial operation, or None if absent.

    Arguments follow get_geo_attr(). Use this when several operations and metadata fields must come from the
    same interface, avoiding collisions with Xarray or Pandas names such as count and shape.
    """

    if hasattr(obj, attr_name):
        return obj
    for accessor_name in accessors:
        accessor = _get_registered_accessor(obj, accessor_name)
        if accessor is not None and hasattr(accessor, attr_name):
            return accessor
    return None


def get_geo_attr(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """Retrieve an attribute from an object, or one of its accessors."""

    # Try direct attribute (Raster, Vector, PointCloud, or accessor class)
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)

    # Try accessors (rst, vct, pc)
    for accessor_name in accessors:
        accessor = _get_registered_accessor(obj, accessor_name)
        if accessor is not None and hasattr(accessor, attr_name):
            return getattr(accessor, attr_name)

    # Fallback
    raise AttributeError(
        f"Attribute '{attr_name}' not found on object {type(obj)} or its potential accessors {accessors}."
    )


def has_geo_attr(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """Check if attribute exists for an object, or one of its accessors."""

    # Check direct attribute (Raster, Vector, PointCloud, or accessor class)
    if hasattr(obj, attr_name):
        return True

    # Check accessors (rst, vct, pc)
    for accessor_name in accessors:
        accessor = _get_registered_accessor(obj, accessor_name)
        if accessor is not None and hasattr(accessor, attr_name):
            return True

    return False


# Helpers for recognizing spatial inputs and selecting their GeoUtils interfaces
#################################################################################


def _get_raster_interface(obj: Any) -> Any:
    """Return the object or rst accessor providing raster coordinate conversion, or None if absent."""

    # Point DataArrays share Xarray's type with rasters, but their one-dimensional coordinates describe rows
    if getattr(obj, "ndim", None) == 1 and _get_pointcloud_interface(obj) is not None:
        return None
    return get_geo_interface(obj, "ij2xy", accessors=("rst",))


def _get_pointcloud_interface(obj: Any) -> Any:
    """Return the object or pc accessor providing ordered point coordinate comparison, or None if absent."""

    return get_geo_interface(obj, "georeferenced_coords_equal", accessors=("pc",))


def _is_raster(obj: Any) -> bool:
    """Check for raster operations on an object or its rst accessor, without requiring a specific class."""

    return _get_raster_interface(obj) is not None


def _is_pointcloud(obj: Any) -> bool:
    """Check for point cloud operations on an object or its pc accessor, without requiring a specific class."""

    return _get_pointcloud_interface(obj) is not None


def _is_vector(obj: Any) -> bool:
    """Check for rasterization on an object or its vct accessor, including point clouds with vector operations."""

    return has_geo_attr(obj, "rasterize", accessors=("vct",))


# Level 0 checks: directly on user input
########################################


def _check_nodata(nodata: Number | None, dtype: DTypeLike | None = None, *, allow_cast: bool = False) -> bool:
    """Helper function to check nodata input and its compatibility with a data type."""

    if nodata is not None and not isinstance(nodata, (int, float, np.integer, np.floating)):
        raise InvalidNodataError("Type of nodata not understood, must be float or int.")
    if nodata is None or dtype is None:
        return True

    # Rasterio may probe an invalid cast before rejecting incompatible nodata
    with np.errstate(invalid="ignore", over="ignore"):
        compatible = bool(rio.dtypes.can_cast_dtype(nodata, dtype))
    if not compatible and not allow_cast:
        raise InvalidNodataError(f"Nodata value {nodata} incompatible with self.dtype {dtype}.")
    return compatible


def _check_resampling(
    method: str,
    allowed: Sequence[str] = get_args(ResamplingMethod),
    name: str = "resampling method",
) -> str:
    """Helper function to check resampling method or backend names."""

    if not isinstance(method, str) or method not in allowed:
        raise InvalidResamplingError(f"Unknown {name}: {method!r}.")
    return method


def _check_valid_data(count: int, minimum: int = 1, message: str = "No finite values are available.") -> None:
    """Check that enough valid observations are available for an operation."""

    if count < minimum:
        raise NoValidDataError(message)


def _check_nodata_propagation(nodata_propagation: str) -> NodataPropagation:
    """
    Check and normalize nodata propagation input.

    :param nodata_propagation: Rule used to handle invalid source values.

    :return: Normalized propagation rule.
    """

    # Lowercase string values so public methods accept the same spelling variants
    normalized = nodata_propagation.lower() if isinstance(nodata_propagation, str) else None
    if normalized not in ("nearest", "ignore", "propagate"):
        raise InvalidNodataError("nodata_propagation must be one of 'nearest', 'ignore' or 'propagate'.")
    return cast("NodataPropagation", normalized)


def _check_crs(crs: Any) -> pyproj.CRS:
    """Function for checking input CRS consistently."""

    # Let Pyproj do the job with from user input
    try:
        crs = pyproj.CRS.from_user_input(crs)
    except pyproj.exceptions.CRSError as e:
        raise InvalidCRSError("Projection not recognized by Pyproj.") from e

    return crs


def _check_point_coordinates(points: Any, *, squeeze: bool = False) -> tuple[NDArrayNum, NDArrayNum]:
    """Helper function to check and normalize point coordinates as X/Y arrays."""

    # Normalize both axes together so invalid arrays receive a spatial input error
    if not isinstance(points, (Sequence, np.ndarray)) or isinstance(points, (str, bytes)):
        raise InvalidPointsError("Expected a sequence of two array-like objects (x, y).")
    if (isinstance(points, np.ndarray) and points.ndim == 0) or len(points) != 2:
        raise InvalidPointsError("Expected a sequence of two array-like objects (x, y).")
    try:
        x, y = (np.asarray(value, dtype=float) for value in points)
    except (TypeError, ValueError, OverflowError) as exception:
        raise InvalidPointsError("Point coordinates must be numeric array-like objects.") from exception

    # Query points may contain NaNs or be empty, but both axes must identify the same locations
    if squeeze:
        x, y = x.squeeze(), y.squeeze()
    if x.shape != y.shape:
        raise InvalidPointsError("Point coordinates must have the same length and shape.")
    return x, y


def _check_bounds(
    bbox: rio.coords.BoundingBox | Sequence[Number] | NDArrayNum | pd.DataFrame | dict[str, Number],
    *,
    allow_empty: bool = True,
) -> tuple[Number, Number, Number, Number]:
    """
    Helper function to check bounds value when provided as a sequence or bounding box object.

    Set allow_empty=False to require positive width and height.
    """

    # If rasterio bounding box, simply extract
    if isinstance(bbox, rio.coords.BoundingBox):
        xmin, ymin, xmax, ymax = bbox.left, bbox.bottom, bbox.right, bbox.top

    # If pandas dataframe (GeoPandas-like), extract if only one row
    elif isinstance(bbox, pd.DataFrame):
        if all(c in bbox.columns for c in ["minx", "maxx", "miny", "maxy"]) and len(bbox) == 1:
            row = bbox.iloc[0]
            xmin, ymin, xmax, ymax = row["minx"], row["miny"], row["maxx"], row["maxy"]
        else:
            raise InvalidBoundsError(
                f"Bounding box as a dataframe must contain columns 'minx', 'maxx', 'miny', "
                f"'maxy' and be of length 1, got columns {bbox.columns} with length {len(bbox)}"
            )

    # If dictionary, check entries
    elif isinstance(bbox, dict):
        required = {"left", "bottom", "right", "top"}
        if set(bbox) != required:
            raise InvalidBoundsError(f"Bounding box as dictionary should have keys {required}, got {set(bbox)}.")

        for k, v in bbox.items():
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
                raise InvalidBoundsError(
                    f"Bounding box dictionary value for {k!r} must be numeric, got {type(v).__name__}."
                )
        xmin, ymin, xmax, ymax = bbox["left"], bbox["bottom"], bbox["right"], bbox["top"]

    # If bbox is an iterable with 4 coordinates (excluding strings and bytes)
    elif isinstance(bbox, (Sequence, np.ndarray)) and not isinstance(bbox, (str, bytes)):
        if isinstance(bbox, np.ndarray) and bbox.ndim != 1:
            raise InvalidBoundsError("Bounding box must be a one-dimensional sequence of length 4.")
        if len(bbox) != 4:
            raise InvalidBoundsError(f"Bounding box must be a sequence of length 4, got a length of {len(bbox)}.")
        xmin, ymin, xmax, ymax = bbox
    else:
        raise InvalidBoundsError(
            f"Cannot interpret bounding box input from object of type {type(bbox).__name__!r}. "
            "Expected a sequence (xmin, ymin, xmax, ymax) or a rasterio BoundingBox."
        )

    # Apply the same numeric and ordering checks to every representation
    if any(isinstance(value, (bool, np.bool_)) for value in (xmin, ymin, xmax, ymax)):
        raise InvalidBoundsError("Bounding box coordinates must be numeric, not booleans.")
    try:
        xmin, ymin, xmax, ymax = (float(value) for value in (xmin, ymin, xmax, ymax))
    except (TypeError, ValueError, OverflowError) as exception:
        raise InvalidBoundsError("Bounding box coordinates must be numeric.") from exception
    if not np.all(np.isfinite((xmin, ymin, xmax, ymax))):
        raise InvalidBoundsError("Bounding box coordinates must be finite.")

    # A point or line can have zero extent, while a target grid needs an area
    comparison = "<=" if allow_empty else "<"
    if xmin > xmax or (not allow_empty and xmin == xmax):
        raise InvalidBoundsError(f"Bounding box xmin must be {comparison} xmax, got xmin={xmin}, xmax={xmax}.")
    if ymin > ymax or (not allow_empty and ymin == ymax):
        raise InvalidBoundsError(f"Bounding box ymin must be {comparison} ymax, got ymin={ymin}, ymax={ymax}.")
    return xmin, ymin, xmax, ymax


def _check_resolution(res: Number | Sequence[Number] | NDArrayNum) -> tuple[Number, Number]:
    """Helper function on checking resolution input."""

    # Case 1: Single scalar resolution
    if isinstance(res, (bool, np.bool_)):
        raise InvalidResolutionError("Resolution must be a number or a sequence of two numbers, not a boolean.")
    if isinstance(res, (float, int, np.floating, np.integer)):
        # Should be finite value
        try:
            res = float(res)
        except (ValueError, OverflowError) as exception:
            raise InvalidResolutionError(f"Resolution must be a finite number, got {res!r}.") from exception
        if not np.isfinite(res):
            raise InvalidResolutionError(f"Resolution must be a finite number, got {res!r}.")
        # Should be strictly positive
        if res <= 0:
            raise InvalidResolutionError(f"Resolution must be strictly positive, got {res!r}.")
        return float(res), float(res)

    # Case 2: Sequence of two numbers for X and Y (xres, yres)
    if isinstance(res, (Sequence, np.ndarray)) and not isinstance(res, (str, bytes)):
        if isinstance(res, np.ndarray) and res.ndim != 1:
            raise InvalidResolutionError("Resolution must be a one-dimensional sequence of two numbers.")
        # Should be a sequence of two
        if len(res) != 2:
            raise InvalidResolutionError(
                f"Resolution must be a number or a sequence of two numbers, got a sequence of length {len(res)}."
            )

        # Should be numeric values
        if any(isinstance(value, (bool, np.bool_)) for value in res):
            raise InvalidResolutionError("Resolution values must be numeric, not booleans.")
        try:
            xres, yres = (float(r) for r in res)
        except (TypeError, ValueError, OverflowError) as e:
            raise InvalidResolutionError("Resolution values must be numeric.") from e

        # Should be finite values
        if not np.isfinite(xres) or not np.isfinite(yres):
            raise InvalidResolutionError("Resolution values must be finite numbers.")

        # Should be strictly positive
        if xres <= 0 or yres <= 0:
            raise InvalidResolutionError(
                f"Resolution values must be strictly positive, got (xres={xres}, yres={yres})."
            )

        return float(xres), float(yres)

    # If none of the above
    raise InvalidResolutionError(
        f"Resolution must be a number or a sequence of two numbers, got object of type {type(res).__name__}"
    )


def _check_shape(shape: Sequence[int | np.integer[Any]] | NDArrayNum, *, allow_empty: bool = True) -> tuple[int, int]:
    """
    Helper function on checking shape input.

    Set allow_empty=False to require positive height and width.
    """

    # Shape should be a sequence of two integers (exclude strings and bytes sequences)
    if not isinstance(shape, (Sequence, np.ndarray)) or isinstance(shape, (str, bytes)):
        raise InvalidShapeError(f"Shape must be a sequence of two integers, got {shape!r}.")
    if isinstance(shape, np.ndarray) and shape.ndim != 1:
        raise InvalidShapeError(f"Shape must be a one-dimensional sequence of two integers, got {shape!r}.")
    if len(shape) != 2:
        raise InvalidShapeError(f"Shape must have length 2, got length {len(shape)}.")

    # Reject fractional sizes and booleans before converting NumPy integer scalars
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in shape):
        raise InvalidShapeError(f"Shape values must be integers, got {shape!r}.")
    nrows, ncols = (int(value) for value in shape)

    # Should be positive
    if nrows < 0 or ncols < 0:
        raise InvalidShapeError(f"Shape values must be non-negative, got {shape!r}.")
    if not allow_empty and (nrows == 0 or ncols == 0):
        raise InvalidShapeError(f"Target grid shape must contain two positive integers, got {shape!r}.")

    return nrows, ncols


def _as_1d_numeric_array(obj: Any, name: str) -> NDArrayNum:
    """Helper to check 1d numeric array-like objects."""

    # Normalize to numeric array if supported.
    try:
        arr = np.asarray(obj, dtype=float)
    except (TypeError, ValueError, OverflowError) as e:
        raise InvalidGridError(f"{name} coordinates must be numeric and array-like.") from e

    # Remove potential 1-size dimensions
    arr = arr.squeeze()

    # Check number of dimensions
    if arr.ndim > 1:
        raise InvalidGridError(f"{name} coordinates must be 1D, got shape {arr.shape}.")

    # Check size (at least 2 to define a grid)
    if arr.size < 2:
        raise InvalidGridError(f"{name} coordinates must contain at least 2 points, got size {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise InvalidGridError(f"{name} grid coordinates must be finite.")

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
    with np.errstate(over="ignore", invalid="ignore"):
        dx = np.diff(x)
        dy = np.diff(y)

    if not np.all(np.isfinite(dx)) or not np.all(np.isfinite(dy)):
        raise InvalidGridError("Grid coordinate spacing must be finite.")
    if not (np.allclose(dx, dx[0]) and np.allclose(dy, dy[0])):
        raise InvalidGridError("Grid coordinates must be regular (equally spaced independently along x and y).")
    if not ((np.all(dx > 0) or np.all(dx < 0)) and (np.all(dy > 0) or np.all(dy < 0))):
        raise InvalidGridError("Grid coordinates must be strictly increasing or decreasing along each axis.")

    return (x, y), (dx[0], dy[0])


# Checks for matching inputs
########################################


def _check_same_crs(
    crs: Any,
    other_crs: Any,
    message: str = "Input coordinate reference systems differ.",
    *,
    ignore_axis_order: bool = False,
) -> None:
    """Helper function to check that two CRSs match, optionally ignoring axis order."""

    normalized_crs = _check_crs(crs) if crs is not None else None
    normalized_other = _check_crs(other_crs) if other_crs is not None else None
    if ignore_axis_order and normalized_crs is not None and normalized_other is not None:
        same_crs = normalized_crs.equals(normalized_other, ignore_axis_order=True)
    else:
        same_crs = crs == other_crs
    if not same_crs:
        raise CRSMismatchError(message)


def _check_same_shape(
    shape: Sequence[int], other_shape: Sequence[int], message: str = "Input array shapes differ."
) -> None:
    """Helper function to check that two array shapes match."""

    if tuple(shape) != tuple(other_shape):
        raise ShapeMismatchError(message)


def _check_same_grid(raster: Any, other: Any, message: str = "Input raster grids differ.") -> None:
    """Helper function to check that two rasters have the same grid without loading pixels."""

    raster_interface = _get_raster_interface(raster)
    other_interface = _get_raster_interface(other)
    if raster_interface is None or other_interface is None:
        raise InvalidGridError("Grid comparison requires two rasters.")
    if not raster_interface.georeferenced_grid_equal(other_interface):
        raise GridMismatchError(message)


def _check_same_points(
    points: Any, other: Any, message: str = "Input point coordinates or their order differ."
) -> None:
    """
    Check that two point inputs have the same ordered coordinates.

    For point clouds, compare the horizontal CRS and X/Y coordinates using georeferenced_coords_equal().
    Coordinate arrays are compared exactly and must already use the same axes and CRS.
    """

    point_interface = _get_pointcloud_interface(points)
    other_interface = _get_pointcloud_interface(other)
    if point_interface is not None and other_interface is not None:
        same_points = point_interface.georeferenced_coords_equal(other_interface)
        if not same_points:
            # Horizontal CRS differences prevent comparing point locations
            crs = _check_crs(point_interface.crs).to_2d() if point_interface.crs is not None else None
            other_crs = _check_crs(other_interface.crs).to_2d() if other_interface.crs is not None else None
            _check_same_crs(crs, other_crs, message)
    else:
        same_points = np.array_equal(points, other)
    if not same_points:
        raise PointsMismatchError(message)


# Level 1 checks: Can accept a reference object to match or manual input
########################################################################


def _check_match_points(
    src: RasterBase | Vector,
    points: tuple[NDArrayNum, NDArrayNum] | tuple[Number, Number] | PointCloudLike,
) -> tuple[tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum], bool]:
    """
    Function for checking and normalizing input of match feature for points consistently.

    :param src: Source object (raster, vector, point cloud).
    :param points: Points object (tuple of arrays/numbers, or point cloud).

    :return: Tuple of point coordinates (X and Y), Boolean to know if inputs were scalar.
    """

    interface = _get_pointcloud_interface(points)
    if getattr(interface, "_is_xr", False):
        x, y, _ = interface.to_xyz()
        coordinates = reproject_points((np.asarray(x), np.asarray(y)), in_crs=interface.crs, out_crs=src.crs)
        return coordinates, False

    # If points implements "bounds" and "crs"
    if has_geo_attr(points, "geometry") and has_geo_attr(points, "crs"):
        crs = get_geo_attr(points, "crs")
        point_geometry = points.geometry  # type: ignore[union-attr]
        try:
            point_x, point_y = point_geometry.x.values, point_geometry.y.values
        except ValueError as exception:
            raise InvalidPointsError("Point reference geometry must contain only points.") from exception
        pts = reproject_points(
            (point_x, point_y),
            in_crs=crs,
            out_crs=src.crs,
        )
        input_scalar = False

        logging.debug(
            f"Match points input: using reference object of type {type(src).__name__!r} "
            f" that implements 'geometry' and 'crs'."
        )

    # Or if points is a sequence of arrays (excluding strings and bytes)
    elif isinstance(points, Sequence) and not isinstance(points, (str, bytes)):
        # Check that they are numeric, 1-dimension, same-length array-like
        x_arr, y_arr = _check_point_coordinates(points, squeeze=True)

        # If inputs are scalars
        if x_arr.ndim == 0 and y_arr.ndim == 0:
            x_arr = x_arr.reshape(1)
            y_arr = y_arr.reshape(1)
            input_scalar = True
        else:
            input_scalar = False

        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise InvalidPointsError(
                "Point coordinates must be a sequence of two 1-dimensional array-like objects, "
                f"got dimensions of {x_arr.ndim} and {y_arr.ndim}."
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
    src: Any,
    bbox: RasterLike | VectorLike | rio.coords.BoundingBox | tuple[Number, Number, Number, Number],
) -> tuple[Number, Number, Number, Number]:
    """Function for checking and normalizing input of match feature on bounds consistently.

    :param src: Source object (raster, vector, point cloud) that has a bbox attribute.
    :param bbox: Bounding box object (tuple or bounding box or raster, vector, point cloud).

    :return: Tuple of bounding box (xmin, ymin, xmax, ymax).
    """

    # Prefer the common bbox name while accepting bounds from third-party geospatial objects
    has_bbox = has_geo_attr(bbox, "bbox")
    has_bounds = has_geo_attr(bbox, "bounds")
    if (has_bbox or has_bounds) and has_geo_attr(bbox, "crs"):
        bbox_attr = "bbox" if has_bbox else "bounds"
        bounds = _check_bounds(get_geo_attr(bbox, bbox_attr))
        crs = _check_crs(get_geo_attr(bbox, "crs"))
        xmin, ymin, xmax, ymax = _get_bounds_projected(bounds=bounds, in_crs=crs, out_crs=src.crs)

        logging.debug(
            f"Match bbox input: using reference object of type {type(src).__name__!r} "
            f"that implements '{bbox_attr}' and 'crs'."
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
            "or a geospatial object implementing 'bbox' and 'crs' such as a raster, vector or point cloud."
        )

    return xmin, ymin, xmax, ymax


def _clip_geodataframe(mask: Any, target_crs: rio.crs.CRS | pyproj.CRS | None) -> gpd.GeoDataFrame:
    """Return clipping geometries as a concrete GeoDataFrame in the target CRS."""

    # Wrap simple geometry inputs without combining their individual parts
    if isinstance(mask, BaseGeometry):
        return gpd.GeoDataFrame(geometry=[mask], crs=target_crs)

    if isinstance(mask, Sequence) and not isinstance(mask, (str, bytes)):
        xmin, ymin, xmax, ymax = _check_bounds(mask)
        return gpd.GeoDataFrame(geometry=[box(xmin, ymin, xmax, ymax)], crs=target_crs)

    # Extract tabular geometries without dissolving features that chunks can select independently
    if isinstance(mask, gpd.GeoSeries):
        dataframe = gpd.GeoDataFrame(geometry=mask)
    elif isinstance(mask, gpd.GeoDataFrame):
        dataframe = gpd.GeoDataFrame(geometry=mask.geometry, crs=mask.crs)
    else:
        interface = get_geo_interface(mask, "_dataset", accessors=("vct", "pc"))
        if interface is not None:
            dataframe = interface._dataset
        else:
            # Footprint as a mask
            raster_interface = get_geo_interface(mask, "footprint", accessors=("rst",))
            if raster_interface is None:
                raise TypeError(
                    "Clipping geometry must be a vector, point cloud, raster, Shapely geometry, "
                    "GeoPandas object or bounding box."
                )
            return _clip_geodataframe(raster_interface.footprint, target_crs=target_crs)

    # Materialize optional Dask inputs once before checking or changing their CRS
    if is_dask_dataframe(dataframe):
        dataframe = dataframe.compute()
    if (
        dataframe.crs is not None
        and target_crs is not None
        and pyproj.CRS.from_user_input(dataframe.crs) != pyproj.CRS.from_user_input(target_crs)
    ):
        dataframe = dataframe.to_crs(target_crs)
    elif dataframe.crs is None and target_crs is not None:
        dataframe = dataframe.set_crs(target_crs)
    return gpd.GeoDataFrame(geometry=dataframe.geometry, crs=dataframe.crs)


def _clip_geometry(mask: Any, target_crs: rio.crs.CRS | pyproj.CRS | None) -> BaseGeometry:
    """Return one clipping geometry normalized and in the target CRS."""

    # Vector clipping needs one mask, while raster clipping can keep features separate for spatial partitioning
    return _clip_geodataframe(mask, target_crs=target_crs).geometry.union_all()


def _get_reproject_crs(
    ref: RasterLike | VectorLike | None = None,
    crs: rio.crs.CRS | pyproj.CRS | str | int | None = None,
) -> rio.crs.CRS:
    """Resolve a target CRS from exactly one reference object or explicit CRS."""

    # Require one way of defining the target CRS
    if (ref is not None and crs is not None) or (ref is None and crs is None):
        raise GeoUtilsInputError("Either of `ref` or `crs` must be set. Not both.")

    # Read the CRS from a geospatial reference when one is provided
    if ref is not None:
        if has_geo_attr(ref, "crs"):
            crs = get_geo_attr(ref, "crs")
        else:
            raise TypeError("Match-reference input must have a 'crs' attribute, such as a raster or vector.")

    # Normalize an explicit CRS through Rasterio, matching the existing transformation interfaces
    _check_crs(crs)
    return rio.crs.CRS.from_user_input(crs)


def _grid_from_bounds_res(
    bounds: tuple[Number, Number, Number, Number], res: Number | tuple[Number, Number]
) -> tuple[tuple[int, int], rio.Affine]:
    """Helper function to check and get grid transform and shape from bounds and resolution."""

    # Check and normalize input bounds
    xmin, ymin, xmax, ymax = _check_bounds(bounds, allow_empty=False)
    # Check and normalize resolution
    res = _check_resolution(res)

    # Derive shape and transform
    shape = round((ymax - ymin) / res[1]), round((xmax - xmin) / res[0])
    shape = _check_shape(shape, allow_empty=False)
    transform = rio.transform.from_origin(xmin, ymax, res[0], res[1])
    return shape, transform


def _grid_from_bounds_shape(
    bounds: tuple[Number, Number, Number, Number], shape: tuple[int, int]
) -> tuple[tuple[int, int], rio.Affine]:
    """Helper function to get grid transform from bounds and shape."""

    xmin, ymin, xmax, ymax = _check_bounds(bounds, allow_empty=False)
    shape = _check_shape(shape, allow_empty=False)
    # Careful, shape inverted (width, height in Rasterio; height, width in NumPy)
    transform = rio.transform.from_bounds(xmin, ymin, xmax, ymax, shape[1], shape[0])
    return shape, transform


def _grid_from_src(
    dst_crs: pyproj.CRS,
    src: Any,
    shape: tuple[int, int] | None = None,
    res: tuple[Number, Number] | Number | None = None,
    bounds: tuple[Number, Number, Number, Number] | None = None,
    georeferencing: dict[str, Any] | None = None,
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
        # Careful, height/width are inverted in Rasterio (width comes first)
        height, width = _check_shape(shape, allow_empty=False)
    else:
        height, width = None, None
    if res is not None:
        res = _check_resolution(res)
    else:
        res = None
    if bounds is not None:
        bounds = _check_bounds(bounds, allow_empty=False)
    else:
        bounds = None

    # Build a complete destination grid without reading point or vector bounds
    if bounds is not None and not hasattr(src, "res"):
        if res is not None:
            return _grid_from_bounds_res(bounds, res)
        elif shape is not None:
            return _grid_from_bounds_shape(bounds, shape)

    # Prefer bbox for GeoUtils objects while accepting bounds from third-party objects
    src_bbox_attr = "bbox" if has_geo_attr(src, "bbox") else "bounds"
    src_bbox = _check_bounds(get_geo_attr(src, src_bbox_attr))

    # First, for a raster source, if all are the same, return exactly source transform and size
    # (to avoid approximation errors from calculations below)
    if georeferencing is None and hasattr(src, "transform") and isinstance(src.transform, rio.Affine):
        if (
            (dst_crs == src.crs)
            & ((shape is None) | (shape == src.shape))
            & ((res is None) | (res == src.res))
            & ((bounds is None) | (bounds == src_bbox))
        ):
            return src.shape, src.transform

    # If there is no input grid (i.e. no resampling involved), just build output grid directly from user inputs
    if not hasattr(src, "res"):
        if bounds is None:
            bounds = _get_bounds_projected(src_bbox, in_crs=src.crs, out_crs=dst_crs)
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

    # Bounds and GCP/RPC models are mutually exclusive in Rasterio
    source_location = (
        georeferencing
        if georeferencing is not None
        else {"left": src_bbox[0], "bottom": src_bbox[1], "right": src_bbox[2], "top": src_bbox[3]}
    )

    # We let Rasterio figure out the default transform (i.e. resolution) in the new CRS with same size output
    transform, width, height = rio.warp.calculate_default_transform(
        src.crs,
        dst_crs,
        src_width,
        src_height,
        resolution=res,  # Only defined if shape is None
        dst_width=width,  # Only defined if res is None
        dst_height=height,  # Only defined if res is None
        **source_location,
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
    dx, dy = abs(float(dxdy[0])), abs(float(dxdy[1]))
    shape = len(y), len(x)
    return shape, rio.transform.from_origin(
        west=x.min() - dx / 2,
        north=y.max() + dy / 2,
        xsize=dx,
        ysize=dy,
    )


def _check_match_grid(
    src: Any,
    ref: RasterLike | VectorLike | None,
    res: Number | tuple[Number, Number] | None,
    shape: tuple[int, int] | None,
    bounds: tuple[Number, Number, Number, Number] | None,
    coords: tuple[NDArrayNum, NDArrayNum] | None,
    crs: pyproj.CRS | None,
    georeferencing: dict[str, Any] | None = None,
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
            # A GCP/RPC image does not define the regular affine grid required for a destination
            raster_reference = get_geo_interface(ref, "transform", accessors=("rst",))
            _check_affine_georeferencing(raster_reference, "Reference grid")

            # Match the spatial grid even when a native Xarray reference includes a band dimension
            dst_shape = _check_shape(get_geo_attr(ref, "shape")[-2:], allow_empty=False)
            dst_transform = get_geo_attr(ref, "transform")
            dst_crs = get_geo_attr(ref, "crs")
            _check_crs(dst_crs)

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
                _warn(category=IgnoredGridWarning, message=msg)

            # If input has an area_or_point attribute, raise warning if inconsistent with source
            if has_geo_attr(ref, "area_or_point") and hasattr(src, "area_or_point"):
                _cast_pixel_interpretation(src.area_or_point, get_geo_attr(ref, "area_or_point"))

        # If reference only defines a partial grid (vector or point cloud-like)
        elif (has_geo_attr(ref, "bbox") or has_geo_attr(ref, "bounds")) and has_geo_attr(ref, "crs"):
            bbox_attr = "bbox" if has_geo_attr(ref, "bbox") else "bounds"
            dst_bounds = _check_bounds(get_geo_attr(ref, bbox_attr))
            dst_crs = _check_crs(get_geo_attr(ref, "crs"))

            logging.debug(
                f"Match grid input: using reference object of type {type(src).__name__!r} "
                f"that implements only '{bbox_attr}' and 'crs' (vector-like)."
            )

            if res is not None and shape is not None:
                raise InvalidGridError(
                    f"Both 'res' and 'shape' were passed to define the grid resolution alongside object of type"
                    f" {type(ref).__name__!r} defining a bbox and CRS. Only provide one of 'res' or 'shape'."
                )
            if res is None and shape is None:
                # If no resolution was defined but source has one (= it is a raster), fallback on source
                if not hasattr(src, "res"):
                    raise InvalidGridError(
                        f"Reference input from object of type {type(ref).__name__!r} only contains "
                        f"a bbox and CRS, and thus requires a provided resolution 'res' or grid shape 'shape' to "
                        f"define a complete grid, but none was passed and source object of type"
                        f" {type(src).__name__!r} (fallback) has none."
                    )
                else:
                    logging.debug(
                        "Match grid input: no resolution defined alongside reference, fallback on "
                        "resolution of source object."
                    )
                    # We calculate for potential differing CRS and target bounds
                    dst_shape, dst_transform = _grid_from_src(
                        dst_crs=dst_crs, src=src, georeferencing=georeferencing, bounds=dst_bounds
                    )
            if coords is not None:
                msg = (
                    f"Reference input from object of type {type(ref).__name__!r} and provided 'res' or 'shape' already "
                    f"define a complete grid, ignoring inputs 'coords'. Pass only 'ref' and ('res' or 'shape') to "
                    f"silence this warning."
                )
                _warn(category=IgnoredGridWarning, message=msg)

            # Resolution and shape: after the above, one or the other must not be None
            # (Note: Both are already defined in target CRS, so no need for projected calculations)
            if res is not None:
                dst_shape, dst_transform = _grid_from_src(
                    dst_crs=dst_crs, src=src, georeferencing=georeferencing, bounds=dst_bounds, res=res
                )
            elif shape is not None:
                dst_shape, dst_transform = _grid_from_src(
                    dst_crs=dst_crs, src=src, georeferencing=georeferencing, bounds=dst_bounds, shape=shape
                )

        else:
            raise InvalidGridError(
                f"Cannot interpret reference grid from object of type {type(ref).__name__!r}. The reference grid "
                f"should implement either 'transform', 'shape' and 'crs' (raster-like), or 'bbox' and "
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
            bounds is not None or has_geo_attr(src, "bbox") or has_geo_attr(src, "bounds")
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
                    dst_shape, dst_transform = _grid_from_src(
                        dst_crs=dst_crs, src=src, georeferencing=georeferencing, bounds=bounds, res=res
                    )
                if shape is not None and bounds is not None:
                    logging.debug("Match grid input: using bounds and shape to derive grid.")
                    dst_shape, dst_transform = _grid_from_src(
                        dst_crs=dst_crs, src=src, georeferencing=georeferencing, bounds=bounds, shape=shape
                    )

                # Fallback to source if res/shape or bounds undefined
                if bounds is None and (shape is not None or res is not None):
                    logging.debug("Match grid input: no bounds defined, fallback on source object.")
                    dst_shape, dst_transform = _grid_from_src(
                        dst_crs=dst_crs, src=src, georeferencing=georeferencing, shape=shape, res=res
                    )
                if bounds is not None and (res is None and shape is None):
                    logging.debug("Match grid input: no resolution defined, fallback on source object.")
                    dst_shape, dst_transform = _grid_from_src(
                        dst_crs=dst_crs, src=src, georeferencing=georeferencing, bounds=bounds
                    )
                if bounds is None and (res is None and shape is None):
                    dst_shape, dst_transform = _grid_from_src(dst_crs=dst_crs, src=src, georeferencing=georeferencing)

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
                _warn(category=IgnoredGridWarning, message=msg)

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
