# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Cosampling of two geospatial datasets and auxiliary values aligned to either input.

Raster and point cloud methods use this module to identify one common spatial support, combine finite data validity
with user masks, then sample every value at the same locations. Independent sampling remains an operation for the
algorithm level in downstream packages because its two outputs do not share locations.

Raster outputs retain the selected grid, with one band per value and a common mask. Point outputs retain the
selected geometries and index labels, with one column per value. Shared support selection and value alignment come
first and are also used by grouped statistics, which retains independent validity for each value. Raster and point
co-sampling follow, then a dispatcher connects them to public object and accessor methods.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geoutils._dispatch import is_dask_array, is_dask_dataframe
from geoutils._misc import import_optional
from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.interface.gridding import GriddingMethod
from geoutils.interface.raster_point import _aligned_raster, _mask_on_raster
from geoutils.raster.array import _selected_raster_data
from geoutils.sampling.subsampling import _dask_subsample, _subsample_numpy
from geoutils.vector.base import _as_vector

SupportName = Literal["self", "other"]

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloud
    from geoutils.raster.raster import Raster


###############################
# 1/ SHARED SPATIAL PREPARATION
###############################


def _sampling_support(inputs: Iterable[Any], at: Any | None) -> Any:
    """Choose explicit support or the first point dataset when combining raster and point inputs."""

    # Unwrap selected bands and columns while retaining their objects as possible spatial support
    objects = [value[0] if isinstance(value, tuple) else value for value in inputs]
    if at is None:
        at = objects[0]
        for value in objects:
            pointcloud = value if hasattr(value, "georeferenced_coords_equal") else getattr(value, "pc", None)
            if pointcloud is not None:
                at = pointcloud
                break

    # Return the normalized object interface used by both co-sampling and grouped statistics
    raster = at if hasattr(at, "ij2xy") else getattr(at, "rst", None)
    pointcloud = at if hasattr(at, "georeferenced_coords_equal") else getattr(at, "pc", None)
    if raster is None and pointcloud is None:
        raise TypeError("at must select raster or point cloud support.")
    return raster if raster is not None else pointcloud


def _sampling_specification(source: Any, specification: Any) -> tuple[Any, Any]:
    """Resolve a caller selection or an external spatial object with an optional band or column."""

    # Interpret ordinary band numbers and column names relative to the calling object
    if specification is None or isinstance(specification, (str, int, np.integer)):
        return source, specification

    # Recognize spatial selections without mistaking a tuple of raw values for an object and selector
    if isinstance(specification, tuple) and len(specification) == 2:
        value = specification[0]
        if (
            hasattr(value, "ij2xy")
            or hasattr(value, "georeferenced_coords_equal")
            or getattr(value, "rst", None) is not None
            or getattr(value, "pc", None) is not None
            or _as_vector(value) is not None
        ):
            return specification
    return specification, None


def _vector_values_at_points(points: gpd.GeoDataFrame, features: gpd.GeoDataFrame, values: NDArrayNum) -> pd.Series:
    """Assign vector feature values to points, with later features winning at overlaps."""

    # Use positional indexes because input point and feature indexes can contain duplicate labels
    left = gpd.GeoDataFrame(geometry=points.geometry.reset_index(drop=True), crs=points.crs)
    right = gpd.GeoDataFrame({"value": values}, geometry=features.geometry.reset_index(drop=True), crs=features.crs)
    if right.crs != left.crs:
        right = right.to_crs(left.crs)
    matches = gpd.sjoin(left, right, how="inner", predicate="intersects").sort_values("index_right")

    # Keep unmatched points missing and preserve their original ordering
    output = np.full(len(points), np.nan)
    output[matches.index.to_numpy()] = matches["value"].to_numpy()
    return pd.Series(output, index=points.index, name="value")


def _sample_vector_values(vector: Any, values: NDArrayNum, support: Any, support_dataframe: Any | None) -> Any:
    """Evaluate numeric vector attributes on a raster grid or ordered point support."""

    # Rasterize feature indexes so missing coverage does not depend on an attribute nodata sentinel
    if hasattr(support, "ij2xy"):
        indexes = np.arange(1, len(values) + 1)
        raster = vector.rasterize(ref=support, in_value=indexes.tolist(), out_value=0, out_dtype=np.int32)
        codes = _selected_raster_data(raster).astype(np.int64)
        return np.take(np.concatenate(([np.nan], values)), codes)

    # Apply the same spatial join independently to point partitions
    if support_dataframe is None:
        raise RuntimeError("Point support coordinates were not prepared.")
    features = vector.ds.compute() if is_dask_dataframe(vector.ds) else vector.ds
    if is_dask_dataframe(support_dataframe):
        sampled = support_dataframe.map_partitions(
            _vector_values_at_points, features, values, meta=pd.Series([], dtype=float, name="value")
        )
        return sampled.to_dask_array(lengths=True)
    return _vector_values_at_points(support_dataframe, features, values).to_numpy()


def _values_at_support(
    source: Any,
    selector: int | str | None,
    *,
    owner: Any,
    support: Any,
    support_dataframe: Any | None,
    name: str,
    interpolation: str,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    preserve_lazy: bool = False,
    strict_owner: bool = False,
) -> Any:
    """Read one selected raster or point value on the requested spatial support."""

    # Normalize geospatial objects and accessors before interpreting raw arrays
    source_raster = source if hasattr(source, "ij2xy") else getattr(source, "rst", None)
    source_pointcloud = (
        source
        if hasattr(source, "georeferenced_coords_equal") and hasattr(source, "data_column")
        else getattr(source, "pc", None)
    )
    support_is_raster = hasattr(support, "ij2xy")

    # Evaluate selected vector attributes without imposing common finite validity
    vector = _as_vector(source) if source_raster is None and source_pointcloud is None else None
    if vector is not None:
        dataframe = vector.ds.compute() if is_dask_dataframe(vector.ds) else vector.ds
        if selector is None or selector not in dataframe.columns:
            raise ValueError("Vector values require an explicit feature column.")
        if not pd.api.types.is_numeric_dtype(dataframe[selector]):
            raise TypeError("Selected vector values must be numeric.")
        return _sample_vector_values(vector, np.asarray(dataframe[selector], dtype=float), support, support_dataframe)

    # Use raw grids directly when their shape already identifies the selected support
    raw_values = source.data if isinstance(source, xr.DataArray) else source
    raw_ndim = raw_values.ndim if hasattr(raw_values, "ndim") else np.asarray(raw_values).ndim
    owner_raster = owner if hasattr(owner, "ij2xy") else getattr(owner, "rst", None)
    if source_raster is None and source_pointcloud is None and raw_ndim >= 2 and owner_raster is not None:
        support_shape = tuple(support.shape) if support_is_raster else None
        direct_values = raw_values.data if isinstance(raw_values, xr.DataArray) else raw_values
        if raw_ndim == 3 and direct_values.shape[0] == 1:
            direct_values = direct_values[0]
        if (
            support_shape is not None
            and tuple(direct_values.shape) == support_shape
            and owner_raster.georeferenced_grid_equal(support)
        ):
            if np.ma.isMaskedArray(direct_values):
                direct_values = np.where(np.ma.getmaskarray(direct_values), np.nan, np.ma.getdata(direct_values))
            return direct_values

        # Attach other raw grids to their owner before applying an explicit reprojection
        source_raster = _aligned_raster(source, owner, support, name, align)

    if source_raster is not None:
        if selector is not None and not isinstance(selector, (int, np.integer)):
            raise TypeError(f"Raster selector for {name!r} must be a band number.")
        band = 1 if selector is None else int(selector)
        raster = _aligned_raster(source_raster, source_raster, support, name, align)
        if support_is_raster:
            return _selected_raster_data(raster, band)

        # Interpolate raster values only after the point support coordinates are known
        if support_dataframe is None:
            raise RuntimeError("Point support coordinates were not prepared.")
        points = (
            support_dataframe
            if is_dask_dataframe(support_dataframe)
            else (support_dataframe.geometry.x.to_numpy(), support_dataframe.geometry.y.to_numpy())
        )
        return raster.interp_points(
            points=points,
            method=interpolation,
            band=band,
            as_array=True,
            mp_config=mp_config,
        )

    # Reject irregular point values on a regular grid because gridding requires an explicit method
    if source_pointcloud is not None:
        if support_is_raster:
            raise ValueError(f"Point value {name!r} cannot be evaluated on raster support without gridding.")
        if source_pointcloud.crs != support.crs:
            if align != "reproject":
                raise ValueError(f"Point value {name!r} does not share the support CRS.")
            source_pointcloud = source_pointcloud.reproject(crs=support.crs)
            source_pointcloud = (
                source_pointcloud if hasattr(source_pointcloud, "georeferenced_coords_equal") else source_pointcloud.pc
            )
        if source_pointcloud is not support and not support.georeferenced_coords_equal(source_pointcloud):
            raise ValueError(f"Point value {name!r} does not share the ordered support coordinates.")

        # Keep point partitions lazy when the caller can reduce their values independently
        dataframe = source_pointcloud.ds
        if is_dask_dataframe(dataframe) and not preserve_lazy:
            dataframe = dataframe.compute()
        column = source_pointcloud.data_column if selector is None else selector
        if column is not None and (not isinstance(column, str) or column not in dataframe.columns):
            raise ValueError(f"Point column {column!r} selected for {name!r} does not exist.")
        values = dataframe.geometry.z if column is None else dataframe[column]
        return values.to_dask_array(lengths=True) if is_dask_dataframe(dataframe) else np.asarray(values)

    # Accept raw point values directly when the selected support supplies their complete ordering
    if support_is_raster:
        raise ValueError(f"Raw value {name!r} cannot be tied to the selected spatial support.")
    if strict_owner:
        owner_pointcloud = owner if hasattr(owner, "georeferenced_coords_equal") else getattr(owner, "pc", None)
        if owner_pointcloud is None or not support.georeferenced_coords_equal(owner_pointcloud):
            raise ValueError(f"One-dimensional value {name!r} must be tied to the selected point support.")
    if np.ma.isMaskedArray(source):
        source = np.where(np.ma.getmaskarray(source), np.nan, np.ma.getdata(source))
    array = (
        source.squeeze() if preserve_lazy and is_dask_array(source) else np.atleast_1d(np.asanyarray(source).squeeze())
    )
    if support_dataframe is None or array.ndim != 1 or len(array) != len(support_dataframe):
        raise ValueError(f"Raw point value {name!r} must contain one value per support point.")
    return array


#####################
# 2/ RASTER SUPPORT
#####################


def _sample_grid_indices(
    valid: Any,
    *,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
) -> tuple[NDArrayNum, NDArrayNum]:
    """Sample common finite cells from eager or lazy validity data."""

    # Delegate lazy selection to the chunk aware sampler before collecting indexes
    if is_dask_array(valid):
        indexes = _dask_subsample(
            valid,
            subsample=subsample,
            return_indices=True,
            random_state=random_state,
            strategy=strategy,
        )
        return tuple(np.asarray(index.compute(), dtype=np.int64) for index in indexes)  # type: ignore[return-value]

    # Encode eager validity as finite values to reuse the established NumPy sampler
    sampling_values = np.where(np.asarray(valid, dtype=bool), 1.0, np.nan)
    indexes = _subsample_numpy(
        sampling_values,
        subsample=subsample,
        return_indices=True,
        random_state=random_state,
        strategy=strategy,
    )
    return tuple(np.asarray(index, dtype=np.int64) for index in indexes)  # type: ignore[return-value]


def _cosample_on_raster(
    first: Any,
    second: Any,
    *,
    support: Any,
    band: int,
    other_band: int,
    auxiliary: Mapping[str, Any],
    auxiliary_bands: Mapping[str, int],
    auxiliary_owners: Mapping[str, Any],
    mask: Any | None,
    mask_mode: str,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    grid_method: GriddingMethod,
    grid_kwargs: Mapping[str, Any],
    align: Literal["raise", "reproject"],
) -> Raster | xr.DataArray:
    """Cosample inputs aligned with one raster grid."""

    # Share spatial preparation with grouped statistics before applying common finite validity
    arrays = {}
    for name, value in {"self": first, "other": second, **auxiliary}.items():
        owner = first if name in {"self", "other"} else auxiliary_owners[name]
        selected_band = band if name == "self" else other_band if name == "other" else auxiliary_bands.get(name, 1)

        # Grid point observations, including plain auxiliary values tied to a point owner
        pointcloud = value if hasattr(value, "georeferenced_coords_equal") else getattr(value, "pc", None)
        owner_points = owner if hasattr(owner, "georeferenced_coords_equal") else getattr(owner, "pc", None)
        if pointcloud is None and owner_points is not None and not hasattr(value, "ij2xy"):
            raw = value.data if isinstance(value, xr.DataArray) else value
            if np.ndim(raw) == 1:
                if np.ma.isMaskedArray(raw):
                    raw = np.where(np.ma.getmaskarray(raw), np.nan, np.ma.getdata(raw))
                copied = owner_points.copy(new_array=raw)
                pointcloud = copied if hasattr(copied, "georeferenced_coords_equal") else copied.pc
        if pointcloud is not None:
            if pointcloud.crs != support.crs:
                if align != "reproject":
                    raise ValueError(f"Point value {name!r} does not share the support CRS.")
                projected = pointcloud.reproject(crs=support.crs)
                pointcloud = projected if hasattr(projected, "georeferenced_coords_equal") else projected.pc
            value = pointcloud.grid(ref=support, resampling=grid_method, **grid_kwargs)
            selected_band = 1

        # Read each converted value on the exact target grid before forming the common mask
        arrays[name] = _values_at_support(
            value,
            selected_band,
            owner=owner,
            support=support,
            support_dataframe=None,
            name=name,
            interpolation="linear",
            align=align,
            mp_config=None,
        )

    # Combine user masking and finite values before drawing any locations
    valid = _mask_on_raster(mask, support, mask_mode, align)
    for array in arrays.values():
        valid = valid & np.isfinite(array)

    # Keep the common grid mask when all finite cells are requested
    if subsample == 1:
        selected = valid
        has_valid = valid.any()
        if not bool(has_valid.compute() if is_dask_array(has_valid) else has_valid):
            raise ValueError("There is no finite data common to all cosampled values.")
    else:
        # Draw locations once and apply the same selection to every output band
        rows, columns = _sample_grid_indices(valid, subsample=subsample, random_state=random_state, strategy=strategy)
        if rows.size == 0:
            raise ValueError("There is no finite data common to all cosampled values.")
        if is_dask_array(valid):
            import_optional("dask")
            import dask.array as da

            grid_rows = da.arange(valid.shape[0], chunks=valid.chunks[0])[:, None]
            grid_columns = da.arange(valid.shape[1], chunks=valid.chunks[1])[None, :]
            selected = da.isin(grid_rows * valid.shape[1] + grid_columns, rows * valid.shape[1] + columns)
        else:
            selected = np.zeros(valid.shape, dtype=bool)
            selected[rows, columns] = True

    # Compare small windows to warn about auxiliaries that can greatly reduce the sample
    if auxiliary:
        base_fraction = float(np.mean(np.isfinite(np.asarray(arrays["self"][:512, :512]))))
        for name in auxiliary:
            auxiliary_fraction = float(np.mean(np.isfinite(np.asarray(arrays[name][:512, :512]))))
            if base_fraction > 0 and auxiliary_fraction < 0.5 * base_fraction:
                warnings.warn(f"Auxiliary variable {name!r} has substantially fewer finite values than 'self'.")

    # Store values in a fixed band order while keeping Dask data lazy
    data = np.stack([np.where(selected, array, np.nan) for array in arrays.values()])
    tags = {"long_name": tuple(arrays)}
    if getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False):
        from geoutils.raster.xr_accessor import RasterAccessor

        return RasterAccessor.from_array(
            data, support.transform, support.crs, nodata=np.nan, area_or_point=support.area_or_point, tags=tags
        )

    # Return the base Raster class because combined values need not describe the calling subclass
    from geoutils.raster.raster import Raster

    data = data.compute() if is_dask_array(data) else data
    return Raster.from_array(
        data, support.transform, support.crs, nodata=np.nan, area_or_point=support.area_or_point, tags=tags
    )


####################
# 3/ POINT SUPPORT
####################


def _raster_valid_at_points(
    raster: Any,
    points: tuple[NDArrayNum, NDArrayNum],
    resample_method: str,
    band: int,
    resample_kwargs: Mapping[str, Any],
) -> NDArrayBool:
    """Evaluate raster validity before choosing the bounded value sample."""

    # Convert finite source cells to a lightweight layer for interpolation
    data = _selected_raster_data(raster, band)
    validity = np.where(np.isfinite(data), 1.0, np.nan).astype(np.float32)

    # Build one validity band even when the source accessor contains multiple bands
    validity_raster = raster.from_array(
        data=validity,
        transform=raster.transform,
        crs=raster.crs,
        nodata=np.nan,
        area_or_point=raster.area_or_point,
    )

    # Normalize accessor outputs before using the common interpolation method
    validity_accessor = validity_raster if hasattr(validity_raster, "ij2xy") else getattr(validity_raster, "rst", None)
    if validity_accessor is None:
        raise TypeError("Could not create a raster validity layer.")

    # Interpolate validity first so rejected points never trigger value reads
    values = validity_accessor.interp_points(
        points=points,
        method=resample_method,
        as_array=True,
        **{"dist_nodata_spread": 0, **resample_kwargs},
    )
    return np.isfinite(np.asarray(values).squeeze())


def _cosample_on_points(
    first: Any,
    second: Any,
    *,
    support: Any,
    band: int,
    other_band: int,
    auxiliary: Mapping[str, Any],
    auxiliary_bands: Mapping[str, int],
    auxiliary_owners: Mapping[str, Any],
    mask: Any | None,
    mask_mode: str,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    resample_method: str,
    resample_kwargs: Mapping[str, Any],
    align: Literal["raise", "reproject"],
) -> PointCloud | gpd.GeoDataFrame:
    """Cosample values native to points and rasters at one ordered point support."""

    # Materialize support coordinates because the bounded output uses their original indexes
    dataframe = support.ds
    dataframe = dataframe.compute() if is_dask_dataframe(dataframe) else dataframe
    x, y = dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()
    points = (x, y)

    # Separate point values from rasters because only rasters require interpolation
    point_values: dict[str, NDArrayNum] = {}
    rasters: dict[str, tuple[Any, int]] = {}
    all_values = {"self": first, "other": second, **auxiliary}
    for name, value in all_values.items():
        value_raster = value if hasattr(value, "ij2xy") else getattr(value, "rst", None)
        value_pointcloud = (
            value
            if hasattr(value, "georeferenced_coords_equal") and hasattr(value, "data_column")
            else getattr(value, "pc", None)
        )

        # Resolve spatial ownership only for raw arrays that carry no metadata
        owner = value
        if value_raster is None and value_pointcloud is None:
            owner = first if name in {"self", "other"} else auxiliary_owners[name]

        # Use array dimensionality to distinguish raw grids from raw point values
        array = value.data if isinstance(value, xr.DataArray) else value
        ndim = array.ndim if hasattr(array, "ndim") else np.asarray(array).ndim
        if value_raster is not None or (value_pointcloud is None and ndim == 2):
            selected_band = band if name == "self" else other_band if name == "other" else auxiliary_bands.get(name, 1)
            rasters[name] = (_aligned_raster(value, owner, support, name, align), selected_band)
        else:
            point_values[name] = _values_at_support(
                value,
                None,
                owner=owner,
                support=support,
                support_dataframe=dataframe,
                name=name,
                interpolation=resample_method,
                align=align,
                mp_config=None,
                strict_owner=True,
            )

    # Combine finite point values with raster validity before selecting support indexes
    valid = np.ones(len(dataframe), dtype=bool)
    for values in point_values.values():
        valid &= np.isfinite(values)
    for raster, selected_band in rasters.values():
        valid &= _raster_valid_at_points(raster, points, resample_method, selected_band, resample_kwargs)

    # Evaluate vector masks on points and raster masks with nearest interpolation
    if mask is not None:
        mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
        vector = _as_vector(mask) if mask_raster is None else None

        # Apply the mask according to its native spatial representation
        if vector is not None:
            mask_values = np.asarray(vector.create_mask(ref=support, as_array=True), dtype=bool).squeeze()
            valid &= mask_values if mask_mode == "inside" else ~mask_values
        elif mask_raster is not None:
            mask_raster = _aligned_raster(mask, mask, support, "mask", align)
            mask_values = mask_raster.interp_points(points=points, method="nearest", as_array=True)
            valid &= np.isfinite(mask_values).squeeze() & (np.asarray(mask_values).squeeze() != 0)
        else:
            # Require raw masks to follow the ordered point support exactly
            mask_values = np.atleast_1d(np.asanyarray(mask).squeeze())
            if np.ma.isMaskedArray(mask_values):
                mask_values = mask_values.filled(False)
            if mask_values.ndim != 1 or len(mask_values) != len(valid) or mask_values.dtype != bool:
                raise ValueError("A point support mask must be Boolean with one value per point.")
            valid &= mask_values

    # Stop before sampling when no location is valid for every requested value
    if not np.any(valid):
        raise ValueError("There is no finite data common to all cosampled values.")

    # Select bounded indexes before interpolating the potentially expensive raster values
    (indices,) = _subsample_numpy(
        np.where(valid, 1.0, np.nan),
        subsample=subsample,
        return_indices=True,
        random_state=random_state,
    )
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    selected_points = (x[indices], y[indices])

    # Extract point values directly and interpolate rasters only at selected coordinates
    sampled = {name: values[indices] for name, values in point_values.items()}
    for name, (raster, selected_band) in rasters.items():
        sampled[name] = np.atleast_1d(
            np.asarray(
                raster.interp_points(
                    points=selected_points,
                    method=resample_method,
                    band=selected_band,
                    as_array=True,
                    **resample_kwargs,
                )
            ).squeeze()
        )

    # Remove interpolation failures while keeping every sampled array aligned
    final_valid = np.ones(len(indices), dtype=bool)
    for values in sampled.values():
        final_valid &= np.isfinite(values)
    indices = indices[final_valid]
    sampled = {name: values[final_valid] for name, values in sampled.items()}

    # Retain selected geometries and their original labels, including duplicate indexes and Z coordinates
    if len(indices) == 0:
        raise ValueError("There is no finite data common to all cosampled values.")
    columns = {name: sampled[name] for name in all_values}
    geometry = dataframe.geometry.iloc[indices].rename("geometry")
    output = gpd.GeoDataFrame(columns, index=dataframe.index[indices], geometry=geometry, crs=support.crs)
    output.attrs["data_column"] = "self"

    # Match the calling interface while using the same column layout for objects and accessors
    if getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False):
        return output
    from geoutils.pointcloud.pointcloud import PointCloud

    # Both primary columns are explicit even when the support also carries a Z coordinate
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Overriding 3D points with with data column 'self'", category=UserWarning
        )
        return PointCloud(output, data_column="self")


############################
# 4/ OBJECT METHOD DISPATCH
############################


def _cosample(
    first: Any,
    second: Any,
    *,
    band: int,
    other_band: int,
    auxiliary: Mapping[str, Any] | None,
    auxiliary_bands: Mapping[str, int] | None,
    auxiliary_at: SupportName | Mapping[str, SupportName] | None,
    at: SupportName | Any | None,
    mask: Any | None,
    mask_mode: Literal["inside", "outside"],
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    raster_point_mode: Literal["grid_points", "resample_raster"] | None,
    grid_method: GriddingMethod,
    resample_method: str,
    grid_kwargs: Mapping[str, Any] | None,
    resample_kwargs: Mapping[str, Any] | None,
    align: Literal["raise", "reproject"],
) -> Raster | PointCloud | xr.DataArray | gpd.GeoDataFrame:
    """Implementation shared by raster and point cloud object methods."""

    # Validate controls before inspecting or loading any spatial inputs
    if second is None:
        raise TypeError("cosample requires an 'other' primary dataset.")
    if mask_mode not in {"inside", "outside"}:
        raise ValueError("mask_mode must be 'inside' or 'outside'.")
    if strategy not in {"sequential", "topk"}:
        raise ValueError("strategy must be 'sequential' or 'topk'.")
    if align not in {"raise", "reproject"}:
        raise ValueError("align must be 'raise' or 'reproject'.")
    if not isinstance(subsample, (int, float)) or subsample <= 0:
        raise ValueError("subsample must be a positive number.")
    if raster_point_mode not in {None, "grid_points", "resample_raster"}:
        raise ValueError("raster_point_mode must be 'grid_points', 'resample_raster' or None.")

    # Keep target locations and method selection under the public co-sampling controls
    grid_kwargs = {} if grid_kwargs is None else dict(grid_kwargs)
    resample_kwargs = {} if resample_kwargs is None else dict(resample_kwargs)
    if {"ref", "grid_coords", "res", "shape", "bounds", "resampling"}.intersection(grid_kwargs):
        raise ValueError("Use at and grid_method to choose gridding locations and method, outside grid_kwargs.")
    if {"points", "method", "band", "as_array", "input_latlon", "return_interpolator"}.intersection(resample_kwargs):
        raise ValueError("Use at, band and resample_method outside resample_kwargs; point coordinates follow at.")

    # Copy auxiliary mappings so later normalization cannot mutate caller state
    auxiliary = {} if auxiliary is None else dict(auxiliary)
    auxiliary_bands = {} if auxiliary_bands is None else dict(auxiliary_bands)
    if any(not isinstance(name, str) or not name for name in auxiliary):
        raise ValueError("Auxiliary names must be non-empty strings.")
    if {"self", "other", "geometry"}.intersection(auxiliary):
        raise ValueError("Auxiliary names cannot be 'self', 'other' or 'geometry'.")
    if not set(auxiliary_bands).issubset(auxiliary):
        raise ValueError("auxiliary_bands contains a name that is not present in auxiliary.")

    # Resolve native support once for auxiliary arrays that lack spatial metadata
    auxiliary_owners: dict[str, Any] = {}
    for name, value in auxiliary.items():
        value_raster = value if hasattr(value, "ij2xy") else getattr(value, "rst", None)
        value_pointcloud = (
            value
            if hasattr(value, "georeferenced_coords_equal") and hasattr(value, "data_column")
            else getattr(value, "pc", None)
        )
        if value_raster is not None or value_pointcloud is not None:
            auxiliary_owners[name] = value
            continue

        # Require raw arrays to identify which primary supplies their spatial support
        owner_name = auxiliary_at.get(name) if isinstance(auxiliary_at, Mapping) else auxiliary_at
        if owner_name is None:
            raise ValueError(f"auxiliary_at must identify the native support of array auxiliary {name!r}.")
        if owner_name not in {"self", "other"}:
            raise ValueError("auxiliary_at values must be 'self' or 'other'.")
        auxiliary_owners[name] = first if owner_name == "self" else second

    # Resolve an explicit target before letting the conversion mode choose a default input
    if isinstance(at, str):
        if at not in {"self", "other"}:
            raise ValueError("at must be 'self', 'other' or a geospatial support object.")
        at = first if at == "self" else second
    if at is None and raster_point_mode is not None:
        candidates = []
        for value in (first, second):
            attribute = "ij2xy" if raster_point_mode == "grid_points" else "georeferenced_coords_equal"
            accessor = "rst" if raster_point_mode == "grid_points" else "pc"
            candidate = value if hasattr(value, attribute) else getattr(value, accessor, None)
            if candidate is not None:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise ValueError("The conversion mode requires one unambiguous input support; select at explicitly.")
        at = candidates[0]
    support = _sampling_support((first, second), at)

    # Normalize object accessors once before dispatching to the support workflow
    raster_support = support if hasattr(support, "ij2xy") else getattr(support, "rst", None)
    point_support = (
        support
        if hasattr(support, "georeferenced_coords_equal") and hasattr(support, "data_column")
        else getattr(support, "pc", None)
    )
    # Reject conflicting directions and reserve neighborhood reduction until its integration is revised
    if (raster_support is not None and raster_point_mode == "resample_raster") or (
        point_support is not None and raster_point_mode == "grid_points"
    ):
        raise ValueError("raster_point_mode conflicts with the grid or point locations selected by at.")
    if point_support is not None and resample_method == "reduce":
        raise NotImplementedError(
            "Window reduction in cosample awaits revision of Raster.reduce_points(); "
            "use reduce_points separately in the meantime."
        )
    if raster_support is not None:
        # Retain the selected raster grid and mask every output band at the same cells
        return _cosample_on_raster(
            first,
            second,
            support=raster_support,
            band=band,
            other_band=other_band,
            auxiliary=auxiliary,
            auxiliary_bands=auxiliary_bands,
            auxiliary_owners=auxiliary_owners,
            mask=mask,
            mask_mode=mask_mode,
            subsample=subsample,
            random_state=random_state,
            strategy=strategy,
            grid_method=grid_method,
            grid_kwargs=grid_kwargs,
            align=align,
        )
    if point_support is not None:
        # Use ordered point coordinates to evaluate both point and raster values
        return _cosample_on_points(
            first,
            second,
            support=point_support,
            band=band,
            other_band=other_band,
            auxiliary=auxiliary,
            auxiliary_bands=auxiliary_bands,
            auxiliary_owners=auxiliary_owners,
            mask=mask,
            mask_mode=mask_mode,
            subsample=subsample,
            random_state=random_state,
            resample_method=resample_method,
            resample_kwargs=resample_kwargs,
            align=align,
        )
    raise TypeError("at must select a raster or point cloud support.")
