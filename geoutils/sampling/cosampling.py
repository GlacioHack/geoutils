# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Sample several geospatial datasets at the same locations."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from geoutils._dispatch import get_geo_attr, has_geo_attr, is_dask_array, is_dask_dataframe
from geoutils._misc import import_optional
from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.interface.gridding import GriddingMethod
from geoutils.raster.array import _selected_raster_data
from geoutils.sampling.subsampling import _dask_subsample, _subsample_numpy
from geoutils.vector.base import _as_geodataframe

if TYPE_CHECKING:
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.pointcloud import PointCloud
    from geoutils.raster.raster import Raster


#################################
# 1/ SHARED SUPPORT AND VALUES
#################################


def _raster_from_input(value: Any, owner: Any, name: str) -> Any:
    """Return a raster input or attach an owner's metadata to a raw array."""

    # Reuse raster objects and accessors so their georeferencing stays authoritative
    raster = value if hasattr(value, "ij2xy") else getattr(value, "rst", None)
    if raster is not None:
        return raster

    # Require a raster owner before interpreting a raw array as gridded data
    owner_raster = owner if hasattr(owner, "ij2xy") else getattr(owner, "rst", None)
    if owner_raster is None:
        raise ValueError(f"Two-dimensional value {name!r} must be tied to a raster input.")

    # Unwrap Xarray and accept the common singleton band representation
    array = value.data if isinstance(value, xr.DataArray) else value
    array = array if hasattr(array, "ndim") else np.asarray(array)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or tuple(array.shape) != tuple(owner_raster.shape):
        raise ValueError(f"Array {name!r} must match the shape of its native raster input.")

    # Attach the owner's grid so later alignment follows the raster API
    return owner_raster.from_array(
        data=array if is_dask_array(array) else np.ma.masked_invalid(array),
        transform=owner_raster.transform,
        crs=owner_raster.crs,
        nodata=owner_raster.nodata,
        area_or_point=owner_raster.area_or_point,
    )


def _aligned_raster(value: Any, owner: Any, support: Any, name: str, align: str) -> Any:
    """Return a raster aligned to raster or point support."""

    # Normalize raw arrays before comparing their owner's spatial reference
    raster = _raster_from_input(value, owner, name)
    if hasattr(support, "georeferenced_grid_equal"):
        if support.georeferenced_grid_equal(raster):
            return raster

        # Reproject grid inputs only when the caller permits spatial alignment
        if align == "reproject":
            return raster.reproject(ref=support, silent=True)
        raise ValueError(f"Raster value {name!r} does not share the selected support grid.")

    # Match a point support CRS without imposing a raster grid
    if raster.crs != support.crs:
        if align != "reproject":
            raise ValueError(f"Raster value {name!r} does not share the point support CRS.")
        raster = raster.reproject(crs=support.crs, silent=True)

    # Normalize reprojection outputs that expose the raster API through an accessor
    normalized = raster if hasattr(raster, "ij2xy") else getattr(raster, "rst", None)
    if normalized is None:
        raise TypeError(f"Raster value {name!r} could not be normalized after reprojection.")
    return normalized


def _mask_on_raster(mask: Any | None, support: Any, mask_mode: str, align: str) -> Any:
    """Evaluate a user mask on raster support."""

    # Keep every cell eligible when no additional mask was requested
    if mask is None:
        return np.ones(support.shape, dtype=bool)

    # Apply vector masks only after excluding raster objects and accessors
    mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)
    if mask_raster is None and has_geo_attr(mask, "create_mask", accessors=("vct",)):
        create_mask = get_geo_attr(mask, "create_mask", accessors=("vct",))
        values = np.asarray(create_mask(ref=support, as_array=True), dtype=bool)
        return values if mask_mode == "inside" else ~values

    # Align raster masks while accepting raw boolean arrays on the support grid
    if mask_raster is not None:
        mask_raster = _aligned_raster(mask_raster, mask_raster, support, "mask", align)
        values = _selected_raster_data(mask_raster, fill_value=False)
    else:
        values = mask if hasattr(mask, "ndim") else np.asarray(mask)
        if np.ma.isMaskedArray(values):
            values = np.ma.asarray(values).filled(False)

    # Drop only a singleton band so one row or one column remains a spatial dimension
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]

    # Reject ambiguous numeric masks and arrays on a different grid shape
    if tuple(values.shape) != tuple(support.shape) or not np.issubdtype(values.dtype, np.bool_):
        raise ValueError("A raster support mask must be boolean and match the support grid.")
    return values


def _sampling_support(inputs: Iterable[Any], at: Any | None) -> Any:
    """Choose the grid or point locations shared by all requested values."""

    # Keep the data object from each `(object, band or column)` request
    objects = [value[0] if isinstance(value, tuple) else value for value in inputs]
    if at is None:
        at = objects[0]
        for value in objects:
            pointcloud = value if hasattr(value, "georeferenced_coords_equal") else getattr(value, "pc", None)
            if pointcloud is not None:
                at = pointcloud
                break

    # Return the Raster or PointCloud interface used by both cosample() and grouped_stats()
    raster = at if hasattr(at, "ij2xy") else getattr(at, "rst", None)
    pointcloud = at if hasattr(at, "georeferenced_coords_equal") else getattr(at, "pc", None)
    if raster is None and pointcloud is None:
        raise TypeError("at must select raster or point cloud support.")
    return raster if raster is not None else pointcloud


def _sampling_specification(source: Any, specification: Any) -> tuple[Any, Any]:
    """Split a value request into its data object and optional band or column."""

    # Interpret ordinary band numbers and column names relative to the calling object
    if specification is None or isinstance(specification, (str, int, np.integer)):
        return source, specification

    # Treat a pair as `(object, selection)` only when its first item carries spatial information
    if isinstance(specification, tuple) and len(specification) == 2:
        value = specification[0]
        if (
            hasattr(value, "ij2xy")
            or hasattr(value, "georeferenced_coords_equal")
            or getattr(value, "rst", None) is not None
            or getattr(value, "pc", None) is not None
            or has_geo_attr(value, "rasterize", accessors=("vct",))
        ):
            return specification
    return specification, None


def _vector_values_at_points(points: gpd.GeoDataFrame, features: gpd.GeoDataFrame, values: NDArrayNum) -> pd.Series:
    """Assign vector feature values to points, with later features winning at overlaps."""

    # Use row positions because point and feature labels may contain duplicates
    left = gpd.GeoDataFrame(geometry=points.geometry.reset_index(drop=True), crs=points.crs)
    right = gpd.GeoDataFrame({"value": values}, geometry=features.geometry.reset_index(drop=True), crs=features.crs)
    if right.crs != left.crs:
        right = right.to_crs(left.crs)
    matches = gpd.sjoin(left, right, how="inner", predicate="intersects").sort_values("index_right")

    # Keep unmatched points missing and preserve their original ordering
    output = np.full(len(points), np.nan)
    output[matches.index.to_numpy()] = matches["value"].to_numpy()
    return pd.Series(output, index=points.index, name="value")


def _sample_vector_values(
    dataframe: gpd.GeoDataFrame, values: NDArrayNum, support: Any, support_dataframe: Any | None
) -> Any:
    """Place numeric feature values on the chosen grid or point locations."""

    # Rasterize feature numbers first so zero always means that no feature covers the cell
    if hasattr(support, "ij2xy"):
        indexes = np.arange(1, len(values) + 1)
        rasterize = get_geo_attr(dataframe, "rasterize", accessors=("vct",))
        raster = rasterize(ref=support, in_value=indexes.tolist(), out_value=0, out_dtype=np.int32)
        codes = _selected_raster_data(raster).astype(np.int64)
        return np.take(np.concatenate(([np.nan], values)), codes)

    # Apply the same GeoPandas join to every Dask dataframe part
    if support_dataframe is None:
        raise RuntimeError("Point support coordinates were not prepared.")
    if is_dask_dataframe(support_dataframe):
        sampled = support_dataframe.map_partitions(
            _vector_values_at_points, dataframe, values, meta=pd.Series([], dtype=float, name="value")
        )
        return sampled.to_dask_array(lengths=True)
    return _vector_values_at_points(support_dataframe, dataframe, values).to_numpy()


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
    """Read one requested value at every location chosen for the final result."""

    # Find the raster or point cloud interface before handling plain arrays
    source_raster = source if hasattr(source, "ij2xy") else getattr(source, "rst", None)
    source_pointcloud = (
        source
        if hasattr(source, "georeferenced_coords_equal") and hasattr(source, "data_column")
        else getattr(source, "pc", None)
    )
    support_is_raster = hasattr(support, "ij2xy")

    # Place vector values now; the parent workflow decides later how to handle missing values
    is_vector = source_raster is None and source_pointcloud is None and has_geo_attr(
        source, "rasterize", accessors=("vct",)
    )
    if is_vector:
        dataframe = _as_geodataframe(source)
        if selector is None or selector not in dataframe.columns:
            raise ValueError("Vector values require an explicit feature column.")
        if not pd.api.types.is_numeric_dtype(dataframe[selector]):
            raise TypeError("Selected vector values must be numeric.")
        return _sample_vector_values(dataframe, np.asarray(dataframe[selector], dtype=float), support, support_dataframe)

    # Use a plain grid directly when its shape and coordinates already match the output grid
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

        # Attach any other plain grid to its owner so _aligned_raster() can reproject it
        source_raster = _aligned_raster(source, owner, support, name, align)

    if source_raster is not None:
        if selector is not None and not isinstance(selector, (int, np.integer)):
            raise TypeError(f"Raster selector for {name!r} must be a band number.")
        band = 1 if selector is None else int(selector)
        raster = _aligned_raster(source_raster, source_raster, support, name, align)
        if support_is_raster:
            return _selected_raster_data(raster, band)

        # Read raster values at the chosen point coordinates
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

    # Reject point values on a raster grid here because only cosample() provides a gridding method
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

        # Keep Dask dataframe parts lazy when grouped_stats() can calculate from them directly
        dataframe = source_pointcloud.ds
        if is_dask_dataframe(dataframe) and not preserve_lazy:
            dataframe = dataframe.compute()
        column = source_pointcloud.data_column if selector is None else selector
        if column is not None and (not isinstance(column, str) or column not in dataframe.columns):
            raise ValueError(f"Point column {column!r} selected for {name!r} does not exist.")
        values = dataframe.geometry.z if column is None else dataframe[column]
        return values.to_dask_array(lengths=True) if is_dask_dataframe(dataframe) else np.asarray(values)

    # Accept a plain 1D array when its owner proves that it follows the chosen point order
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


##################
# 2/ RASTER OUTPUT
##################


def _sample_grid_indices(
    valid: Any,
    *,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
) -> tuple[NDArrayNum, NDArrayNum]:
    """Choose row and column numbers from the cells available in every input."""

    # Let the Dask sampler choose cells before loading their row and column numbers
    if is_dask_array(valid):
        indexes = _dask_subsample(
            valid,
            subsample=subsample,
            return_indices=True,
            random_state=random_state,
            strategy=strategy,
        )
        return tuple(np.asarray(index.compute(), dtype=np.int64) for index in indexes)  # type: ignore[return-value]

    # Turn available cells into finite values accepted by the shared NumPy sampler
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
    """Build a raster whose bands contain values sampled at the same grid cells."""

    # Put every requested value on the output grid before deciding which cells to keep
    arrays = {}
    for name, value in {"self": first, "other": second, **auxiliary}.items():
        owner = first if name in {"self", "other"} else auxiliary_owners[name]
        selected_band = band if name == "self" else other_band if name == "other" else auxiliary_bands.get(name, 1)

        # Grid point observations, including plain arrays tied to a point cloud
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

        # Read the selected band after every value uses the output grid
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

    # Keep only cells allowed by the user mask and available in every value
    valid = _mask_on_raster(mask, support, mask_mode, align)
    for array in arrays.values():
        valid = valid & np.isfinite(array)

    # Keep every available cell when the caller does not request a smaller sample
    if subsample == 1:
        selected = valid
        has_valid = valid.any()
        if not bool(has_valid.compute() if is_dask_array(has_valid) else has_valid):
            raise ValueError("There is no finite data common to all cosampled values.")
    else:
        # Draw cell locations once so every output band uses the same sample
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

    # Warn when an added value appears likely to remove more than half of the available cells
    if auxiliary:
        base_fraction = float(np.mean(np.isfinite(np.asarray(arrays["self"][:512, :512]))))
        for name in auxiliary:
            auxiliary_fraction = float(np.mean(np.isfinite(np.asarray(arrays[name][:512, :512]))))
            if base_fraction > 0 and auxiliary_fraction < 0.5 * base_fraction:
                warnings.warn(f"Auxiliary variable {name!r} has substantially fewer finite values than 'self'.")

    # Build bands in the documented order and keep Dask data lazy
    data = np.stack([np.where(selected, array, np.nan) for array in arrays.values()])
    tags = {"long_name": tuple(arrays)}
    if getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False):
        from geoutils.raster.xr_accessor import RasterAccessor

        return RasterAccessor.from_array(
            data, support.transform, support.crs, nodata=np.nan, area_or_point=support.area_or_point, tags=tags
        )

    # Return a base Raster because the combined bands may no longer describe the caller's subclass
    from geoutils.raster.raster import Raster

    data = data.compute() if is_dask_array(data) else data
    return Raster.from_array(
        data, support.transform, support.crs, nodata=np.nan, area_or_point=support.area_or_point, tags=tags
    )


#################
# 3/ POINT OUTPUT
#################


def _raster_valid_at_points(
    raster: Any,
    points: tuple[NDArrayNum, NDArrayNum],
    resample_method: str,
    band: int,
    resample_kwargs: Mapping[str, Any],
) -> NDArrayBool:
    """Find which output points can receive a value from one raster band."""

    # Mark available source cells with one and missing cells with NaN
    data = _selected_raster_data(raster, band)
    validity = np.where(np.isfinite(data), 1.0, np.nan).astype(np.float32)

    # Build a one-band raster so only the requested source band controls point selection
    validity_raster = raster.from_array(
        data=validity,
        transform=raster.transform,
        crs=raster.crs,
        nodata=np.nan,
        area_or_point=raster.area_or_point,
    )

    # Get the Raster interface when from_array() returns an Xarray object
    validity_accessor = validity_raster if hasattr(validity_raster, "ij2xy") else getattr(validity_raster, "rst", None)
    if validity_accessor is None:
        raise TypeError("Could not create a raster validity layer.")

    # Check point coverage first so we read values only at points that can be kept
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
    """Build a point table whose columns contain values sampled at the same points."""

    # Load the output point coordinates and keep their original row labels
    dataframe = support.ds
    dataframe = dataframe.compute() if is_dask_dataframe(dataframe) else dataframe
    x, y = dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy()
    points = (x, y)

    # Separate point values from rasters because only raster values need interpolation
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

        # Use an owner only for plain arrays that do not carry coordinates
        owner = value
        if value_raster is None and value_pointcloud is None:
            owner = first if name in {"self", "other"} else auxiliary_owners[name]

        # Treat 2D arrays as grids and 1D arrays as values following the point order
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

    # Keep points where every point value is available and every raster can be read
    valid = np.ones(len(dataframe), dtype=bool)
    for values in point_values.values():
        valid &= np.isfinite(values)
    for raster, selected_band in rasters.values():
        valid &= _raster_valid_at_points(raster, points, resample_method, selected_band, resample_kwargs)

    # Read vector and raster masks at the chosen point locations
    if mask is not None:
        mask_raster = mask if hasattr(mask, "ij2xy") else getattr(mask, "rst", None)

        # Use the mask's own geometry, grid, or point order
        if mask_raster is None and has_geo_attr(mask, "create_mask", accessors=("vct",)):
            create_mask = get_geo_attr(mask, "create_mask", accessors=("vct",))
            mask_values = np.asarray(create_mask(ref=support, as_array=True), dtype=bool).squeeze()
            valid &= mask_values if mask_mode == "inside" else ~mask_values
        elif mask_raster is not None:
            mask_raster = _aligned_raster(mask, mask, support, "mask", align)
            mask_values = mask_raster.interp_points(points=points, method="nearest", as_array=True)
            valid &= np.isfinite(mask_values).squeeze() & (np.asarray(mask_values).squeeze() != 0)
        else:
            # Require a plain mask to contain one boolean value per output point
            mask_values = np.atleast_1d(np.asanyarray(mask).squeeze())
            if np.ma.isMaskedArray(mask_values):
                mask_values = mask_values.filled(False)
            if mask_values.ndim != 1 or len(mask_values) != len(valid) or mask_values.dtype != bool:
                raise ValueError("A point support mask must be boolean with one value per point.")
            valid &= mask_values

    # Stop before sampling when no point has every requested value
    if not np.any(valid):
        raise ValueError("There is no finite data common to all cosampled values.")

    # Choose point rows before reading the more expensive raster values
    (indices,) = _subsample_numpy(
        np.where(valid, 1.0, np.nan),
        subsample=subsample,
        return_indices=True,
        random_state=random_state,
    )
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    selected_points = (x[indices], y[indices])

    # Select point columns directly and read raster values only at the chosen points
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

    # Remove points where interpolation still returned NaN from any sampled column
    final_valid = np.ones(len(indices), dtype=bool)
    for values in sampled.values():
        final_valid &= np.isfinite(values)
    indices = indices[final_valid]
    sampled = {name: values[final_valid] for name, values in sampled.items()}

    # Keep the chosen geometries, duplicate row labels, and any Z coordinates
    if len(indices) == 0:
        raise ValueError("There is no finite data common to all cosampled values.")
    columns = {name: sampled[name] for name in all_values}
    geometry = dataframe.geometry.iloc[indices].rename("geometry")
    output = gpd.GeoDataFrame(columns, index=dataframe.index[indices], geometry=geometry, crs=support.crs)
    output.attrs["data_column"] = "self"

    # Return a GeoDataFrame for an accessor call and a PointCloud for an object call
    if getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False):
        return output
    from geoutils.pointcloud.pointcloud import PointCloud

    # Keep both primary values as columns even when the point geometry already has a Z coordinate
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Overriding 3D points with with data column 'self'", category=UserWarning
        )
        return PointCloud(output, data_column="self")


##########################
# 4/ PUBLIC METHOD ROUTING
##########################


def _cosample(
    first: Any,
    second: Any,
    *,
    band: int,
    other_band: int,
    auxiliary: Mapping[str, Any] | None,
    auxiliary_bands: Mapping[str, int] | None,
    auxiliary_at: Literal["self", "other"] | Mapping[str, Literal["self", "other"]] | None,
    at: Literal["self", "other"] | Any | None,
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
    """Check public cosample() inputs and run the raster or point workflow.

    _sampling_support() chooses the shared output locations. _cosample_on_raster() or _cosample_on_points() then
    aligns every requested value, applies one common selection, and builds the matching spatial result.
    """

    # Check options before inspecting or loading any input data
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

    # Keep output locations and method choices in the named public arguments
    grid_kwargs = {} if grid_kwargs is None else dict(grid_kwargs)
    resample_kwargs = {} if resample_kwargs is None else dict(resample_kwargs)
    if {"ref", "grid_coords", "res", "shape", "bounds", "resampling"}.intersection(grid_kwargs):
        raise ValueError("Use at and grid_method to choose gridding locations and method, outside grid_kwargs.")
    if {"points", "method", "band", "as_array", "input_latlon", "return_interpolator"}.intersection(resample_kwargs):
        raise ValueError("Use at, band and resample_method outside resample_kwargs; point coordinates follow at.")

    # Copy added value settings so this function cannot change the caller's dictionaries
    auxiliary = {} if auxiliary is None else dict(auxiliary)
    auxiliary_bands = {} if auxiliary_bands is None else dict(auxiliary_bands)
    if any(not isinstance(name, str) or not name for name in auxiliary):
        raise ValueError("Auxiliary names must be non-empty strings.")
    if {"self", "other", "geometry"}.intersection(auxiliary):
        raise ValueError("Auxiliary names cannot be 'self', 'other' or 'geometry'.")
    if not set(auxiliary_bands).issubset(auxiliary):
        raise ValueError("auxiliary_bands contains a name that is not present in auxiliary.")

    # Record which primary input supplies coordinates for each plain added array
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

        # Require every plain array to name the primary input whose locations it follows
        owner_name = auxiliary_at.get(name) if isinstance(auxiliary_at, Mapping) else auxiliary_at
        if owner_name is None:
            raise ValueError(f"auxiliary_at must identify the native support of array auxiliary {name!r}.")
        if owner_name not in {"self", "other"}:
            raise ValueError("auxiliary_at values must be 'self' or 'other'.")
        auxiliary_owners[name] = first if owner_name == "self" else second

    # Choose explicit output locations before using the raster and point conversion direction
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

    # Find the raster or point cloud interface that defines the output locations
    raster_support = support if hasattr(support, "ij2xy") else getattr(support, "rst", None)
    point_support = (
        support
        if hasattr(support, "georeferenced_coords_equal") and hasattr(support, "data_column")
        else getattr(support, "pc", None)
    )
    # Reject a conversion direction that conflicts with the chosen output locations
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
        # Build output bands on the selected raster grid and give them the same mask
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
        # Build output columns on the selected point coordinates and keep their order
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
