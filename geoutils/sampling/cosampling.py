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

The module first defines the compact result, then handles raster and point support with shared spatial normalization.
The final dispatcher connects those workflows to the public object methods.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import xarray as xr

from geoutils._dispatch import is_dask_array, is_dask_dataframe
from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.interface.raster_point import _aligned_raster, _mask_on_raster
from geoutils.raster.array import _selected_raster_data
from geoutils.sampling.subsampling import _dask_subsample, _subsample_numpy
from geoutils.vector.base import _as_vector

SupportName = Literal["self", "other"]

__all__ = ["CoSampleResult"]


####################
# 1/ COSAMPLE RESULT
####################


@dataclass(frozen=True)
class CoSampleResult:
    """Values from two datasets sampled at common finite locations.

    The result stores only eager arrays at the selected locations. ``indices`` refers to the original support: one
    array for point support and ``(row, column)`` for raster support.

    :param self_values: Values sampled from the object on which ``cosample`` was called.
    :param other_values: Values sampled from the other primary dataset.
    :param auxiliary: Named auxiliary values sampled at the same locations.
    :param coordinates: X and Y coordinates of the selected locations.
    :param indices: Selected point indexes or raster row and column indexes.
    :param support_kind: Native support type, either ``"raster"`` or ``"pointcloud"``.
    :param support_shape: Original raster shape or point count.
    :param crs: Coordinate reference system of the result.
    :param transform: Raster transform when the support is a raster.
    """

    self_values: NDArrayNum
    other_values: NDArrayNum
    auxiliary: Mapping[str, NDArrayNum]
    coordinates: tuple[NDArrayNum, NDArrayNum]
    indices: tuple[NDArrayNum, ...]
    support_kind: Literal["raster", "pointcloud"]
    support_shape: tuple[int, ...]
    crs: Any
    transform: Any | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Copy primary values to give the result independent and aligned storage
        self_values = np.asarray(self.self_values).copy()
        other_values = np.asarray(self.other_values).copy()
        if self_values.ndim != 1 or other_values.ndim != 1 or len(self_values) != len(other_values):
            raise ValueError("Cosampled primary values must be aligned one-dimensional arrays.")

        # Copy associated arrays before checking their alignment against the primaries
        auxiliary = {name: np.asarray(values).copy() for name, values in self.auxiliary.items()}
        if any(values.ndim != 1 or len(values) != len(self_values) for values in auxiliary.values()):
            raise ValueError("Every auxiliary value must align with the primary samples.")
        coordinates = tuple(np.asarray(values).copy() for values in self.coordinates)
        indices = tuple(np.asarray(values, dtype=np.int64).copy() for values in self.indices)
        if len(coordinates) != 2 or any(len(values) != len(self_values) for values in coordinates + indices):
            raise ValueError("Coordinates and support indexes must align with the sampled values.")

        # Validate support metadata so indexes retain one unambiguous meaning
        if self.support_kind not in {"raster", "pointcloud"}:
            raise ValueError("support_kind must be 'raster' or 'pointcloud'.")
        if self.support_kind == "raster" and len(indices) != 2:
            raise ValueError("Raster support requires row and column indexes.")
        if self.support_kind == "pointcloud" and len(indices) != 1:
            raise ValueError("Point support requires one point index array.")

        # Freeze owned arrays to prevent changes that would silently break alignment
        for values in (self_values, other_values, *auxiliary.values(), *coordinates, *indices):
            values.setflags(write=False)
        object.__setattr__(self, "self_values", self_values)
        object.__setattr__(self, "other_values", other_values)
        object.__setattr__(self, "auxiliary", MappingProxyType(auxiliary))
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))

    def __len__(self) -> int:
        """Return the number of common sampled locations."""

        return len(self.self_values)

    def to_pointcloud(self, *, self_name: str = "self", other_name: str = "other") -> Any:
        """Convert selected values and coordinates to a GeoUtils point cloud.

        :param self_name: Column name for values from the calling object.
        :param other_name: Column name for values from the other primary dataset.
        :returns: Point cloud with both primary and all auxiliary value columns.
        """

        # Import the public class lazily to avoid a circular module dependency
        from geoutils.pointcloud.pointcloud import PointCloud

        # Reject duplicate names before constructing the output dataframe
        if self_name == other_name or self_name in self.auxiliary or other_name in self.auxiliary:
            raise ValueError("Primary and auxiliary output column names must be unique.")

        # Build every value column on the same selected point geometry
        columns = {self_name: self.self_values, other_name: self.other_values, **self.auxiliary}
        geometry = gpd.points_from_xy(*self.coordinates, crs=self.crs)
        return PointCloud(gpd.GeoDataFrame(columns, geometry=geometry, crs=self.crs), data_column=self_name)

    def to_arrays(
        self, *, preserve_shape: bool = False
    ) -> tuple[NDArrayNum, NDArrayNum, dict[str, NDArrayNum], tuple[NDArrayNum, NDArrayNum]]:
        """Return primary, auxiliary and coordinate arrays for procedural workflows.

        With ``preserve_shape=True``, values are expanded to the original raster shape or point count and locations
        outside the common sample are filled with NaN. Coordinates always remain limited to selected locations.

        :param preserve_shape: Whether to expand values to the original support.
        :returns: Calling object values, other values, auxiliary mapping and selected X/Y coordinates.
        """

        # Return compact copies when the caller does not need the original support shape
        if not preserve_shape:
            return (
                self.self_values.copy(),
                self.other_values.copy(),
                {name: values.copy() for name, values in self.auxiliary.items()},
                tuple(values.copy() for values in self.coordinates),  # type: ignore[return-value]
            )

        # Expand every value array with NaN so unsampled support remains explicit
        output: list[NDArrayNum] = []
        for values in (self.self_values, self.other_values, *self.auxiliary.values()):
            dtype = values.dtype if np.issubdtype(values.dtype, np.floating) else np.dtype("float64")
            expanded = np.full(self.support_shape, np.nan, dtype=dtype)
            expanded[self.indices] = values
            output.append(expanded)

        # Rebuild the auxiliary mapping in its original insertion order
        auxiliary = dict(zip(self.auxiliary, output[2:]))
        return (
            output[0],
            output[1],
            auxiliary,
            tuple(values.copy() for values in self.coordinates),  # type: ignore[return-value]
        )

    def to_support(self) -> dict[str, NDArrayNum]:
        """Expand all values to the original support with NaN outside the sample.

        :returns: Mapping containing ``"self"``, ``"other"`` and every auxiliary variable.
        """

        self_values, other_values, auxiliary, _ = self.to_arrays(preserve_shape=True)
        return {"self": self_values, "other": other_values, **auxiliary}


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
    align: str,
) -> CoSampleResult:
    """Cosample inputs aligned with one raster grid."""

    # Reject point primaries because their coordinates define the natural support
    first_pointcloud = (
        first
        if hasattr(first, "georeferenced_coords_equal") and hasattr(first, "data_column")
        else getattr(first, "pc", None)
    )
    second_pointcloud = (
        second
        if hasattr(second, "georeferenced_coords_equal") and hasattr(second, "data_column")
        else getattr(second, "pc", None)
    )
    if first_pointcloud is not None or second_pointcloud is not None:
        raise ValueError("A point cloud primary requires point support.")

    # Align both primary grids before deriving their common finite mask
    first_raster = _aligned_raster(first, first, support, "self", align)
    second_raster = _aligned_raster(second, first, support, "other", align)

    # Attach metadata to raw auxiliaries and align every auxiliary grid once
    auxiliary_rasters: dict[str, Any] = {}
    for name, value in auxiliary.items():
        auxiliary_rasters[name] = _aligned_raster(value, auxiliary_owners[name], support, name, align)

    # Select requested bands without loading lazy raster values
    arrays = {
        "self": _selected_raster_data(first_raster, band),
        "other": _selected_raster_data(second_raster, other_band),
        **{
            name: _selected_raster_data(raster, auxiliary_bands.get(name, 1))
            for name, raster in auxiliary_rasters.items()
        },
    }

    # Combine user masking and finite values before drawing any locations
    valid = _mask_on_raster(mask, support, mask_mode, align)
    for array in arrays.values():
        valid = valid & np.isfinite(array)
    rows, columns = _sample_grid_indices(valid, subsample=subsample, random_state=random_state, strategy=strategy)
    if rows.size == 0:
        raise ValueError("There is no finite data common to all cosampled values.")

    # Compare small windows to warn about auxiliaries that can greatly reduce the sample
    base_fraction = float(np.mean(np.isfinite(np.asarray(arrays["self"][:512, :512]))))
    for name in auxiliary:
        auxiliary_fraction = float(np.mean(np.isfinite(np.asarray(arrays[name][:512, :512]))))
        if base_fraction > 0 and auxiliary_fraction < 0.5 * base_fraction:
            warnings.warn(f"Auxiliary variable {name!r} has substantially fewer finite values than 'self'.")

    # Materialize only selected cells so lazy inputs remain bounded in memory
    sampled: dict[str, NDArrayNum] = {}
    for name, array in arrays.items():
        values = array.vindex[rows, columns].compute() if is_dask_array(array) else array[rows, columns]
        values = values.filled(np.nan) if np.ma.isMaskedArray(values) else values
        sampled[name] = np.asarray(values)

    # Preserve support coordinates and metadata for conversions back to spatial objects
    x, y = support.ij2xy(rows, columns)
    return CoSampleResult(
        self_values=sampled["self"],
        other_values=sampled["other"],
        auxiliary={name: sampled[name] for name in auxiliary},
        coordinates=(np.asarray(x), np.asarray(y)),
        indices=(rows, columns),
        support_kind="raster",
        support_shape=tuple(support.shape),
        crs=support.crs,
        transform=support.transform,
        attrs={"strategy": strategy, "align": align},
    )


####################
# 3/ POINT SUPPORT
####################


def _point_aligned_values(
    value: Any, owner: Any, support: Any, support_dataframe: gpd.GeoDataFrame, name: str, align: str
) -> NDArrayNum:
    """Read one value native to the selected point support."""

    # Normalize point objects and accessors before comparing their coordinates
    pointcloud = (
        value
        if hasattr(value, "georeferenced_coords_equal") and hasattr(value, "data_column")
        else getattr(value, "pc", None)
    )
    if pointcloud is not None:
        # Reproject coordinates only when the caller explicitly permits it
        if pointcloud.crs != support.crs:
            if align != "reproject":
                raise ValueError(f"Point cloud value {name!r} does not share the support CRS.")
            pointcloud = pointcloud.reproject(crs=support.crs)
            pointcloud = pointcloud if hasattr(pointcloud, "georeferenced_coords_equal") else pointcloud.pc

        # Require ordered equality because array values follow their point positions
        if not support.georeferenced_coords_equal(pointcloud):
            raise ValueError(f"Point cloud value {name!r} does not share the ordered support coordinates.")

        # Materialize the bounded point table before extracting its selected data column
        dataframe = pointcloud.ds
        dataframe = dataframe.compute() if is_dask_dataframe(dataframe) else dataframe
        values = dataframe[pointcloud.data_column] if pointcloud.data_column is not None else dataframe.geometry.z
        return np.asarray(values)

    # Tie raw arrays to an owner with the same ordered point coordinates
    owner_pointcloud = (
        owner
        if hasattr(owner, "georeferenced_coords_equal") and hasattr(owner, "data_column")
        else getattr(owner, "pc", None)
    )
    if owner_pointcloud is None or not support.georeferenced_coords_equal(owner_pointcloud):
        raise ValueError(f"One-dimensional value {name!r} must be tied to the selected point support.")

    # Validate the raw array length against the already materialized support
    if np.ma.isMaskedArray(value):
        value = np.where(np.ma.getmaskarray(value), np.nan, np.ma.getdata(value))
    array = np.atleast_1d(np.asarray(value).squeeze())
    if array.ndim != 1 or len(array) != len(support_dataframe):
        raise ValueError(f"Array {name!r} must contain one value per support point.")
    return array


def _raster_valid_at_points(
    raster: Any, points: tuple[NDArrayNum, NDArrayNum], interpolation: str, band: int
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
        method=interpolation,
        dist_nodata_spread=0,
        as_array=True,
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
    interpolation: str,
    align: str,
) -> CoSampleResult:
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
            point_values[name] = _point_aligned_values(value, owner, support, dataframe, name, align)

    # Combine finite point values with raster validity before selecting support indexes
    valid = np.ones(len(dataframe), dtype=bool)
    for values in point_values.values():
        valid &= np.isfinite(values)
    for raster, selected_band in rasters.values():
        valid &= _raster_valid_at_points(raster, points, interpolation, selected_band)

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
    indices = np.asarray(indices, dtype=np.int64)
    selected_points = (x[indices], y[indices])

    # Extract point values directly and interpolate rasters only at selected coordinates
    sampled = {name: values[indices] for name, values in point_values.items()}
    for name, (raster, selected_band) in rasters.items():
        sampled[name] = np.atleast_1d(
            np.asarray(
                raster.interp_points(points=selected_points, method=interpolation, band=selected_band, as_array=True)
            ).squeeze()
        )

    # Remove interpolation failures while keeping every sampled array aligned
    final_valid = np.ones(len(indices), dtype=bool)
    for values in sampled.values():
        final_valid &= np.isfinite(values)
    indices = indices[final_valid]
    sampled = {name: values[final_valid] for name, values in sampled.items()}

    # Preserve original point indexes so outputs can be expanded to their source support
    return CoSampleResult(
        self_values=sampled["self"],
        other_values=sampled["other"],
        auxiliary={name: sampled[name] for name in auxiliary},
        coordinates=(x[indices], y[indices]),
        indices=(indices,),
        support_kind="pointcloud",
        support_shape=(len(dataframe),),
        crs=support.crs,
        attrs={"strategy": "sequential", "align": align, "interpolation": interpolation},
    )


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
    interpolation: str,
    align: Literal["raise", "reproject"],
) -> CoSampleResult:
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

    # Copy auxiliary mappings so later normalization cannot mutate caller state
    auxiliary = {} if auxiliary is None else dict(auxiliary)
    auxiliary_bands = {} if auxiliary_bands is None else dict(auxiliary_bands)
    if any(not isinstance(name, str) or not name for name in auxiliary):
        raise ValueError("Auxiliary names must be non-empty strings.")
    if {"self", "other"}.intersection(auxiliary):
        raise ValueError("Auxiliary names cannot be 'self' or 'other'.")
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

    # Prefer point support automatically because raster values can be evaluated at points
    if at is None:
        point_supports = []
        for value in (first, second):
            pointcloud = (
                value
                if hasattr(value, "georeferenced_coords_equal") and hasattr(value, "data_column")
                else getattr(value, "pc", None)
            )
            if pointcloud is not None:
                point_supports.append(value)
        support = point_supports[0] if point_supports else first
    elif isinstance(at, str):
        # Resolve named support without accepting silent misspellings
        if at not in {"self", "other"}:
            raise ValueError("at must be 'self', 'other' or a geospatial support object.")
        support = first if at == "self" else second
    else:
        support = at

    # Normalize object accessors once before dispatching to the support workflow
    raster_support = support if hasattr(support, "ij2xy") else getattr(support, "rst", None)
    point_support = (
        support
        if hasattr(support, "georeferenced_coords_equal") and hasattr(support, "data_column")
        else getattr(support, "pc", None)
    )
    if raster_support is not None:
        # Keep regular grids in index space until selected values are materialized
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
            interpolation=interpolation,
            align=align,
        )
    raise TypeError("at must select a raster or point cloud support.")
