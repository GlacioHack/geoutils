# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Shared module to sample geospatial inputs on shared spatial supports: grids or point locations.

Especially used by cosampling, but also pair/subsampling for masking.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_pointcloud,
    _is_raster,
    _is_vector,
    get_geo_attr,
    has_geo_attr,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, NDArrayNum
from geoutils.raster.array import _selected_raster_data
from geoutils.vector.base import _as_geodataframe

if TYPE_CHECKING:
    from geoutils.interface.interpolation import InterpolationMethod
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.vector.base import VectorLike

#################################
# 1/ ARRAY VALUES AND INPUT GRIDS
#################################


def _normalize_sampling_input(value: Any) -> Any:
    """Treat Xarray inputs without both x and y coordinates as plain arrays, preserving Dask storage."""

    if isinstance(value, xr.DataArray) and not {"x", "y"}.issubset(value.coords):
        return value.data
    return value


def _as_array(value: Any) -> Any:
    """
    Extract array values while preserving NumPy masks and Dask arrays.

    :param value: Array, Xarray DataArray, or Pandas/Dask Series, Index or DataFrame to unwrap.

    :returns: Array values with existing NumPy masks or Dask storage preserved.
    """

    # Extract values from Xarray and Pandas inputs as NumPy arrays, or Dask arrays for chunked data
    if isinstance(value, xr.DataArray):
        return value.data
    if is_dask_dataframe(value):
        return value.to_dask_array(lengths=True)
    if isinstance(value, (pd.Series, pd.Index, pd.DataFrame)):
        return value.to_numpy(copy=False)
    return value if is_dask_array(value) else np.asanyarray(value)


def _normalize_mask_array(mask: Any | None, shape: tuple[int, ...]) -> Any:
    """
    Return a boolean mask value per location, keeping missing mask entries ineligible.

    The mask comes from _mask_at_support(); this step checks array values after any spatial placement.

    :param shape: Required output shape. The mask must contain the same number of entries and is reshaped to match.

    :returns: Boolean NumPy or Dask array, or None when no mask was supplied.
    """

    # Return None for no mask, to avoid allocating extra memory
    if mask is None:
        return None

    # Ensure lazy arrays and proper missing values in eager masked arrays
    values = _as_array(mask)
    if np.ma.isMaskedArray(values):
        values = values.filled(False)
    elif is_dask_array(values):
        # Apply the same missing-entry rule within lazy blocks without collecting the mask
        import_optional("dask")
        import dask.array as da

        values = da.ma.filled(values, False)
    if not np.issubdtype(values.dtype, np.bool_) or values.size != math.prod(shape):
        raise ValueError("Argument ``mask`` must be boolean and contain one value per input location.")
    return values.reshape(shape)


def _raster_from_input(
    value: RasterLike | ArrayLike, input_support: RasterLike | PointCloudLike, name: str
) -> RasterBase:
    """
    Return a raster input or attach its original grid metadata to a plain array.

    ``value`` is the source passed to _values_at_support(); input_support and name follow that parent.
    Spatial raster inputs use their own metadata. Plain arrays must match the input_support grid.
    """

    # Use the raster's own grid information when the input is already a raster or raster accessor
    value = _normalize_sampling_input(value)
    raster = _get_raster_interface(value)
    if raster is not None:
        return raster

    # Find the raster that supplies coordinates for a plain array; an array alone has no grid information
    input_raster = _get_raster_interface(input_support)
    if input_raster is None:
        raise ValueError(f"Two-dimensional value {name!r} must be tied to a raster input.")

    # Read the array and drop a band dimension of length one, then require the original raster shape
    array: Any = value if hasattr(value, "ndim") else np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or tuple(array.shape) != tuple(input_raster.shape):
        raise ValueError(f"Array {name!r} must match the shape of its native raster input.")

    # Replace masked entries with NaN using NumPy's dtype promotion so large integer values keep their precision
    if np.ma.isMaskedArray(array):
        array = np.where(np.ma.getmaskarray(array), np.nan, np.ma.getdata(array))

    # Keep raw Dask arrays lazy even when their grid comes from a GeoUtils Raster
    from_array = input_raster.from_array
    if is_dask_array(array):
        from geoutils.raster.xr_accessor import RasterAccessor

        from_array = RasterAccessor.from_array

    # Create a raster with the array values and the original grid's coordinates
    # Mark missing array values without applying the original raster's nodata value
    raster = from_array(
        data=array if is_dask_array(array) else np.ma.masked_invalid(array),
        transform=input_raster.transform,
        crs=input_raster.crs,
        nodata=None,
        area_or_point=input_raster.area_or_point,
    )
    return _get_raster_interface(raster)


def _aligned_raster(
    value: RasterLike | ArrayLike,
    input_support: RasterLike | PointCloudLike,
    support: RasterLike | PointCloudLike,
    name: str,
    align: str,
    mp_config: MultiprocConfig | None = None,
) -> RasterBase:
    """
    Return a raster reprojected to the output support.

    Match the complete grid for raster output, or only the coordinate system for point output.

    See _values_at_support() for arguments.
    """

    # Turn plain arrays into rasters using their original grid before checking the output coordinates
    raster = _raster_from_input(value, input_support, name)

    # Use nearest-neighbor resampling for masks so interpolation does not turn booleans into fractions
    resampling = "nearest" if raster.is_mask else None
    if has_geo_attr(support, "georeferenced_grid_equal", ("rst",)):
        if cast("RasterBase", support).georeferenced_grid_equal(raster):
            return raster

        # Reproject to the full output grid when allowed, or report the grid mismatch
        if align == "reproject":
            projected = raster.reproject(ref=support, resampling=resampling, silent=True, mp_config=mp_config)
            return _get_raster_interface(projected)
        raise ValueError(f"Raster value {name!r} does not share the selected support grid.")

    # For point output, match only the CRS because the output points do not define a raster grid
    if raster.crs != support.crs:
        if align != "reproject":
            raise ValueError(f"Raster value {name!r} does not share the point support CRS.")
        raster = raster.reproject(crs=support.crs, resampling=resampling, silent=True, mp_config=mp_config)

    # Return a raster or its accessor so later steps can call the same raster methods
    return _get_raster_interface(raster)


def _mask_on_raster(
    mask: RasterLike | VectorLike | ArrayLike,
    support: RasterBase,
    mask_mode: str,
    align: str,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Evaluate a user mask on raster support.

    See _mask_at_support() for arguments.

    Return one boolean value per support cell. Vector masks use mask_mode, while raster and plain array masks use
    their boolean values directly. The caller handles an absent mask before calling this function.
    """

    # Read Xarray masks without x and y coordinates as plain arrays
    mask = _normalize_sampling_input(mask)
    is_raster = _is_raster(mask)

    # Mark cells inside vector shapes, then invert the mask when outside was requested
    if not is_raster and has_geo_attr(mask, "create_mask", accessors=("vct",)):
        create_mask = get_geo_attr(mask, "create_mask", accessors=("vct",))
        values = create_mask(ref=support, as_array=True, mp_config=mp_config)
        values = _normalize_mask_array(values, tuple(support.shape))
        return values if mask_mode == "inside" else ~values

    # Read raster masks on the output grid; plain arrays already correspond to its cells
    if is_raster:
        mask_raster = _aligned_raster(mask, mask, support, "mask", align, mp_config=mp_config)
        values = _selected_raster_data(mask_raster, fill_value=False)
    else:
        values = _as_array(mask)

    # Remove a band dimension of length one, but preserve rows and columns even when their size is one
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]

    # Require the mask to match the output grid and contain boolean values
    if tuple(values.shape) != tuple(support.shape):
        raise ValueError("A raster support mask must be boolean and match the support grid.")
    return _normalize_mask_array(values, tuple(support.shape))


#################################
# 2/ SUPPORT AND VALUE SELECTION
#################################


@overload
def _sampling_support(inputs: Iterable[Any], at: RasterBase) -> RasterBase: ...


@overload
def _sampling_support(inputs: Iterable[Any], at: PointCloudBase) -> PointCloudBase: ...


@overload
def _sampling_support(inputs: Iterable[Any], at: RasterLike | PointCloudLike | None) -> RasterBase | PointCloudBase: ...


def _sampling_support(inputs: Iterable[Any], at: RasterLike | PointCloudLike | None) -> RasterBase | PointCloudBase:
    """
    Choose the grid or point locations shared by all requested values.

    :param inputs: Ordered raster/point cloud inputs, optionally paired with a band or column selector. With no
        explicit ``at`` passed, use the first point cloud if present, otherwise the first input.
    :param at: Explicit raster or point cloud support, or None. The caller resolves any 'self'/'other' names first.

    :returns: Raster or point cloud interface supplying the chosen output locations.
    """

    at = _normalize_sampling_input(at)

    # Unless at is provided, use the first point cloud's locations, or the first input if no point cloud is present
    if at is None:
        # Take the object from each (object, selector) pair only when input locations determine the support
        objects = [_normalize_sampling_input(value[0] if isinstance(value, tuple) else value) for value in inputs]
        at = objects[0]
        for value in objects:
            pointcloud = _get_pointcloud_interface(value)
            if pointcloud is not None:
                at = pointcloud
                break

    # Return the chosen raster or point cloud, using its accessor when needed
    raster = _get_raster_interface(at)
    pointcloud = _get_pointcloud_interface(at)
    if raster is None and pointcloud is None:
        raise TypeError("Argument ``at`` must select raster or point cloud support.")
    return raster if raster is not None else pointcloud


def _sampling_specification(source: RasterLike | PointCloudLike, specification: Any) -> tuple[Any, Any]:
    """
    Split a value request into its data object and optional band or column.

    :param source: Calling raster or point cloud used when the request contains only a selector.
    :param specification: Band number, column name, separate data object, or (spatial object, selector) pair.
        None selects the calling object's default value.
    :returns: Data object and selector; a separate object with no explicit selection receives a None selector.
    """

    # Use the calling object when the request only specifies a band number or column name
    if specification is None or isinstance(specification, (str, int, np.integer)):
        return source, specification

    # Read (object, band or column) pairs only when the first item is a raster, point cloud or vector
    if isinstance(specification, tuple) and len(specification) == 2:
        value = specification[0]
        if _is_raster(value) or _is_pointcloud(value) or _is_vector(value):
            return specification
    return specification, None


def _vector_values_at_points(points: gpd.GeoDataFrame, features: gpd.GeoDataFrame) -> pd.Series:
    """
    Assign vector feature values to points, with later features winning at overlaps.

    Called by _sample_vector_values(): points is its support_dataframe (or one Dask partition), and features
    contains geometries in the point CRS and their numeric value column. Return a Series in the original point
    order, with NaN outside all features.
    """

    # Number points by row so duplicate index labels cannot mix up their matches
    left = gpd.GeoDataFrame(geometry=points.geometry.reset_index(drop=True), crs=points.crs)

    # Match points to intersecting features in the same CRS
    # Sort matches by feature row so the last overlapping feature supplies the value
    matches = gpd.sjoin(left, features, how="inner", predicate="intersects").sort_values("index_right")

    # Write matched values back in point order and leave unmatched points as NaN
    output = np.full(len(points), np.nan)
    output[matches.index.to_numpy()] = matches["value"].to_numpy()
    return pd.Series(output, index=points.index, name="value")


def _sample_vector_values(
    dataframe: gpd.GeoDataFrame,
    values: NDArrayNum,
    support: RasterBase | PointCloudBase,
    support_dataframe: Any | None,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Place numeric feature values on the chosen grid or point locations.

    Rasterization handles grid alignment and Dask or multiprocessing blocks. Point values use a spatial join,
    partitioned for Dask inputs, with later features winning at overlaps and missing values outside coverage.

    The output support, support_dataframe and mp_config follow _values_at_support().

    :param dataframe: Vector features whose geometries define where each value applies.
    :param values: One numeric value per dataframe row, in the same order.
    :returns: Array on the support grid or ordered support points, with NaN outside all features.
    """

    # Rasterize feature row numbers starting at one, reserving zero for cells outside all features
    if _is_raster(support):
        indexes = np.arange(1, len(values) + 1)
        rasterize = get_geo_attr(dataframe, "rasterize", accessors=("vct",))
        raster = rasterize(ref=support, in_value=indexes.tolist(), out_value=0, out_dtype=np.int32, mp_config=mp_config)

        # Replace feature numbers with their values, and replace zero with NaN
        codes = _selected_raster_data(raster).astype(np.int64)
        return np.take(np.concatenate(([np.nan], values)), codes)

    # Number features by row so duplicate index labels cannot mix up their matches
    if support_dataframe is None:
        raise RuntimeError("Point support coordinates were not prepared.")
    features = gpd.GeoDataFrame(
        {"value": values}, geometry=dataframe.geometry.reset_index(drop=True), crs=dataframe.crs
    )

    # Match feature coordinates to the output CRS once before sampling every point partition
    if features.crs != support_dataframe.crs:
        features = features.to_crs(support_dataframe.crs)

    # Assign feature values to output points, processing each Dask chunk separately when needed
    if is_dask_dataframe(support_dataframe):
        sampled = support_dataframe.map_partitions(
            _vector_values_at_points, features, meta=pd.Series([], dtype=float, name="value")
        )
        return sampled.to_dask_array(lengths=True)
    return _vector_values_at_points(support_dataframe, features).to_numpy()


def _aligned_pointcloud(
    pointcloud: PointCloudBase,
    support: RasterBase | PointCloudBase,
    name: str,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None = None,
) -> PointCloudBase:
    """
    Match point coordinates to the support CRS and, for point output, its ordered XY locations.

    Raster output only requires matching coordinate systems before gridding. Point output also requires identical
    ordered coordinates after any reprojection, so later value selection can use positional arrays.
    Multiprocessing callers provide a temporary configuration kept alive until the aligned points have been read.
    """

    # Match the output CRS before gridding points or comparing their ordered coordinates
    if pointcloud.crs != support.crs:
        if align != "reproject":
            raise ValueError(f"Point value {name!r} does not share the support CRS.")
        point_config = None
        if mp_config is not None:
            # Preserve exact coordinates in a separate point file and adapt raster tiles to row partitions
            point_config = mp_config.copy()
            point_config.driver = "GPKG"
            point_config.outfile += ".gpkg"
            if isinstance(point_config.chunks, tuple):
                point_config.chunks = math.prod(point_config.chunks)
        pointcloud = pointcloud.reproject(crs=support.crs, mp_config=point_config)
        pointcloud = _get_pointcloud_interface(pointcloud)

    # Require the same point coordinates in the same order when output locations are points
    if not _is_raster(support) and pointcloud is not support:
        if not cast("PointCloudBase", support).georeferenced_coords_equal(pointcloud):
            raise ValueError(f"Point value {name!r} does not share the ordered support coordinates.")
    return pointcloud


def _point_values_at_support(
    source: PointCloudBase | ArrayLike,
    selector: int | str | None,
    *,
    support_dataframe: Any | None,
    name: str,
    point_partition_lengths: tuple[int, ...] | None = None,
) -> Any:
    """
    Read an aligned point cloud column or plain array in output point order.

    Point cloud coordinates are prepared by _aligned_pointcloud() before this reader is called. Plain arrays must
    contain one value per support point. Keep Dask storage and reuse known partition lengths when reading the
    support dataframe.
    """

    # Read the requested column, or use Z coordinates when no column is selected
    source_pointcloud = _get_pointcloud_interface(source)
    if source_pointcloud is not None:
        dataframe = source_pointcloud.ds
        column = source_pointcloud.data_column if selector is None else selector
        if column is not None and (not isinstance(column, str) or column not in dataframe.columns):
            raise ValueError(f"Point column {column!r} selected for {name!r} does not exist.")
        values = dataframe.geometry.z if column is None else dataframe[column]

        # Reuse output chunk lengths only when reading from that same dataframe
        if is_dask_dataframe(dataframe):
            lengths = (
                point_partition_lengths
                if dataframe is support_dataframe and point_partition_lengths is not None
                else True
            )
            return values.to_dask_array(lengths=lengths)
        return np.asarray(values)

    # Replace masked entries with NaN and remove extra dimensions around the point values
    point_array = _as_array(source)
    if np.ma.isMaskedArray(point_array):
        point_array = np.where(np.ma.getmaskarray(point_array), np.nan, np.ma.getdata(point_array))
    array = np.atleast_1d(point_array.squeeze())

    # Check the array length against the output point count, using saved Dask chunk lengths when available
    point_count = (
        sum(point_partition_lengths)
        if point_partition_lengths is not None
        else (len(support_dataframe) if support_dataframe is not None else None)
    )
    if array.ndim != 1 or len(array) != point_count:
        raise ValueError(f"Raw point value {name!r} must contain one value per support point.")
    return array


def _values_at_support(
    source: RasterLike | PointCloudLike | VectorLike | ArrayLike,
    selector: int | str | None,
    *,
    input_support: RasterLike | PointCloudLike,
    support: RasterBase | PointCloudBase,
    support_dataframe: Any | None,
    name: str,
    interpolation: InterpolationMethod,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    point_partition_lengths: tuple[int, ...] | None = None,
) -> Any:
    """
    Read one requested value (band of raster or column of point cloud) at every location chosen for the final result.

    _aligned_raster() prepares grids before band selection or interpolation. _aligned_pointcloud() checks point
    locations before _point_values_at_support() reads their values, while _sample_vector_values() assigns feature
    values by location. This parent defines the input and output supports used by the spatial helpers shared by
    co-sampling and statistics.

    :param source: Raster, point cloud, vector, or plain array providing the values to place.
    :param selector: Raster band number (counting from one) or point/vector column name. None uses the first raster
        band or active point value; vectors require an explicit column.
    :param input_support: Raster or point cloud providing the original grid or ordered coordinates associated with
        plain input arrays. Spatial source objects already carry their own locations.
    :param support: Raster or point cloud defining where the result is evaluated, which may differ from input_support.
    :param support_dataframe: Prepared GeoDataFrame or Dask GeoDataFrame in output point order; None for a raster grid.
    :param name: Value label used to identify this input in error messages.
    :param interpolation: Raster interpolation method when evaluating values at points.
    :param align: Whether mismatched grids or coordinate systems raise an error or are reprojected to the support.
    :param mp_config: Optional multiprocessing configuration passed to spatial operations that support it.
    :param point_partition_lengths: Optional known support_dataframe partition lengths, avoiding repeated row counts
        when reading columns from that same table.
    :returns: One selected value per support cell or point, as a NumPy or Dask array.
    """

    # Identify rasters and point clouds first so they are not mistaken for plain arrays or vectors
    source = _normalize_sampling_input(source)
    source_raster = _get_raster_interface(source)
    source_pointcloud = _get_pointcloud_interface(source)
    support_is_raster = _is_raster(support)

    # Read the requested numeric vector column at each output location
    is_vector = source_raster is None and source_pointcloud is None and _is_vector(source)
    if is_vector:
        dataframe = _as_geodataframe(source)
        if selector is None or selector not in dataframe.columns:
            raise ValueError("Vector values require an explicit feature column.")
        if not pd.api.types.is_numeric_dtype(dataframe[selector]):
            raise TypeError("Selected vector values must be numeric.")
        return _sample_vector_values(
            dataframe, np.asarray(dataframe[selector], dtype=float), support, support_dataframe, mp_config=mp_config
        )

    # Inspect plain arrays only after excluding spatial objects, whose values may still be file-backed
    if source_raster is None and source_pointcloud is None:
        direct_values: Any = source
        raw_ndim = np.ndim(direct_values)
        input_raster = _get_raster_interface(input_support)
        if raw_ndim >= 2 and input_raster is not None:
            # Return arrays directly when their shape and original grid already match the output grid
            support_shape = tuple(cast("RasterBase", support).shape) if support_is_raster else None
            if raw_ndim == 3 and direct_values.shape[0] == 1:
                direct_values = direct_values[0]
            if (
                support_shape is not None
                and tuple(direct_values.shape) == support_shape
                and input_raster.georeferenced_grid_equal(support)
            ):
                if np.ma.isMaskedArray(direct_values):
                    direct_values = np.where(np.ma.getmaskarray(direct_values), np.nan, np.ma.getdata(direct_values))
                return direct_values

            # Give other plain arrays their original grid before moving them to the output locations
            source_raster = _raster_from_input(source, input_support, name)

    # Select the requested raster band and match its grid or CRS to the output locations
    if source_raster is not None:
        if selector is not None and not isinstance(selector, (int, np.integer)):
            raise TypeError(f"Raster selector for {name!r} must be a band number.")
        band = 1 if selector is None else int(selector)
        raster = _aligned_raster(source_raster, source_raster, support, name, align, mp_config=mp_config)
        if support_is_raster:
            return _selected_raster_data(raster, band)

        # Interpolate raster values at the output point coordinates
        if support_dataframe is None:
            raise RuntimeError("Point support coordinates were not prepared.")
        points = (
            support_dataframe
            if is_dask_dataframe(support_dataframe)
            else (support_dataframe.geometry.x.to_numpy(), support_dataframe.geometry.y.to_numpy())
        )
        known_partitions = is_dask_dataframe(points) and point_partition_lengths is not None
        values = raster.interp_points(
            points=points,
            method=interpolation,
            band=band,
            as_array=not known_partitions,
            mp_config=mp_config,
        )

        # Reuse known chunk lengths when converting interpolated point columns back to arrays
        if known_partitions:
            return get_geo_attr(values, "data", ("pc",)).to_dask_array(lengths=point_partition_lengths)
        return values

    # Reject point values on a raster grid here because only cosample() provides a gridding method
    if support_is_raster:
        if source_pointcloud is not None:
            raise ValueError(f"Point value {name!r} cannot be evaluated on raster support without gridding.")
        raise ValueError(f"Raw value {name!r} cannot be tied to the selected spatial support.")

    # Read point values after checking that their ordered coordinates match the output support
    with ExitStack() as temporary_files:
        if source_pointcloud is not None:
            intermediate = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
            source = _aligned_pointcloud(source_pointcloud, support, name, align, mp_config=intermediate)
        return _point_values_at_support(
            cast("PointCloudBase | ArrayLike", source),
            selector,
            support_dataframe=support_dataframe,
            name=name,
            point_partition_lengths=point_partition_lengths,
        )


def _mask_at_support(
    mask: RasterLike | VectorLike | ArrayLike | None,
    support: RasterBase | PointCloudBase,
    *,
    support_dataframe: Any | None = None,
    mask_mode: str = "inside",
    align: Literal["raise", "reproject"] = "raise",
    mp_config: MultiprocConfig | None = None,
    point_partition_lengths: tuple[int, ...] | None = None,
) -> Any | None:
    """
    Reproject a boolean mask on raster or point support for sampling and statistics.

    _mask_on_raster() handles raster grids. For point support, create_mask() identifies points inside vector features
    and _values_at_support() aligns raster or point masks. _normalize_mask_array() applies the same boolean
    and missing-value rules to their results.

    A missing mask input stays None to avoid allocating an unused mask.

    :param mask: Boolean array or spatial raster, point cloud or vector mask. None keeps all locations eligible.
    :param support: Raster or point cloud defining the grid or ordered points on which to evaluate the mask.
    :param support_dataframe: Prepared output point table, optionally lazy. If omitted for point support, use its ds.
    :param mask_mode: Keep locations inside or outside vector features. Boolean masks use their values directly.
    :param align: Whether a raster or point mask with a different grid or coordinate system raises or is reprojected.
    :param mp_config: Optional multiprocessing configuration for spatial mask placement.
    :param point_partition_lengths: Optional known support_dataframe partition lengths, as in _values_at_support().
    :returns: Boolean array on the support, optionally lazy, or None when no mask was supplied.
    """

    # 1/ Check mask options and use the raster mask helper when the output is a grid
    if mask is None:
        return None
    if mask_mode not in {"inside", "outside"}:
        raise ValueError("Argument ``mask_mode`` must be 'inside' or 'outside'.")
    if _is_raster(support):
        return _mask_on_raster(mask, cast("RasterBase", support), mask_mode, align, mp_config=mp_config)
    if support_dataframe is None:
        support_dataframe = get_geo_attr(support, "ds", accessors=("pc",))
    mask = _normalize_sampling_input(mask)

    # Check for point clouds before vectors because point clouds also have vector methods
    mask_is_raster = _is_raster(mask)
    mask_is_pointcloud = _is_pointcloud(mask)
    is_vector = not mask_is_raster and not mask_is_pointcloud and _is_vector(mask)

    # 2/ Calculate which output points the mask allows

    # If input is a vector
    if is_vector:
        # Use create_mask() to find points inside vector shapes, excluding their boundaries
        create_mask = get_geo_attr(mask, "create_mask", accessors=("vct",))
        known_partitions = is_dask_dataframe(support_dataframe) and point_partition_lengths is not None
        values = create_mask(ref=support_dataframe, as_array=not known_partitions, mp_config=mp_config)

        # Reuse known chunk lengths when reading the point mask from a Dask dataframe
        if known_partitions:
            values = get_geo_attr(values, "data", ("pc",)).to_dask_array(lengths=point_partition_lengths)
        if mask_mode == "outside":
            values = ~values

    # If input is raster or point cloud
    elif mask_is_raster or mask_is_pointcloud:
        # Require boolean raster or point values so numeric data cannot be mistaken for a mask
        dtype = (
            get_geo_attr(mask, "dtype", accessors=("rst",))
            if mask_is_raster
            else get_geo_attr(mask, "data", accessors=("pc",)).dtype
        )
        is_boolean = np.issubdtype(dtype, np.bool_)
        if mask_is_raster:
            is_boolean |= get_geo_attr(mask, "is_mask", accessors=("rst",))
        if not is_boolean:
            raise ValueError("A point support mask must contain boolean values.")

        # Read mask values at the output points; nearest interpolation preserves boolean raster values
        sampled = _values_at_support(
            mask,
            1 if mask_is_raster else None,
            input_support=mask,
            support=support,
            support_dataframe=support_dataframe,
            name="mask",
            interpolation="nearest",
            align=align,
            mp_config=mp_config,
            point_partition_lengths=point_partition_lengths,
        )

        # Exclude missing mask values as well as False values
        values = np.isfinite(sampled) & (sampled != 0)

    # If input is a boolean array
    else:
        # Check that the plain array has one boolean value for each output point
        count = sum(point_partition_lengths) if point_partition_lengths is not None else len(support_dataframe)
        return _normalize_mask_array(mask, (count,))

    # 3/ Return one boolean value per point without counting the Dask rows again
    return _normalize_mask_array(values, (values.size,))
