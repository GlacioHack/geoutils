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

"""This module contains shared functions for Xarray selection/construction for raster and point cloud accessors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import xarray as xr
from pyproj import CRS
from rioxarray.exceptions import MissingSpatialDimensionError
from xarray.core.utils import dict_equiv

from geoutils._dispatch import is_dask_array
from geoutils._misc import import_optional


def _select_dataset_variables(
    dataset: xr.Dataset, compatible: Sequence[str], variables: Sequence[str] | None
) -> list[str]:
    """Select compatible value variables and raise proper errors otherwise."""

    selected = list(compatible) if variables is None else list(variables)
    if isinstance(variables, str) or not selected or len(selected) != len(set(selected)):
        raise ValueError("Argument 'variables' must be a nonempty sequence of distinct variable names.")
    unknown = [name for name in selected if name not in dataset.data_vars]
    if unknown:
        raise ValueError(f"Unknown Dataset variables: {unknown}.")
    incompatible = [name for name in selected if name not in compatible]
    if incompatible:
        raise ValueError(f"Variables are incompatible with this spatial representation: {incompatible}.")
    return selected


def _same_coordinate(left: xr.Variable, right: xr.Variable) -> bool:
    """Compare coordinate metadata, accounting for eager/lazy."""

    # 1/ Check dimensions and shape (before comparing metadata or values)
    if left.dims != right.dims or left.shape != right.shape:
        return False

    # 2/ Compare attributes, allowing equivalent CRS descriptions on scalar coordinates
    if not dict_equiv(left.attrs, right.attrs):
        if left.dims:
            return False

        # NetCDF mappings can describe a CRS with projection parameters instead of WKT
        references: list[CRS] = []
        for attrs in (left.attrs, right.attrs):
            description = attrs.get("crs_wkt", attrs.get("spatial_ref"))
            if description is not None:
                references.append(CRS.from_user_input(description))
            elif "grid_mapping_name" in attrs:
                references.append(CRS.from_cf(attrs))
            else:
                return False

        # Compare CRS definitions independently of axis order
        source_crs, target_crs = references
        if not source_crs.equals(target_crs, ignore_axis_order=True):
            return False

        # Equivalent WKT spellings and CF projection fields describe the same grid mapping
        fields = set(source_crs.to_cf()) | set(target_crs.to_cf()) | {"spatial_ref", "GeoTransform"}
        left_attributes = {key: value for key, value in left.attrs.items() if key not in fields}
        right_attributes = {key: value for key, value in right.attrs.items() if key not in fields}
        if not dict_equiv(left_attributes, right_attributes):
            return False

        # Compare numerical grid transforms regardless of string formatting
        transforms = [
            tuple(float(value) for value in attrs.get("GeoTransform", "").split())
            for attrs in (left.attrs, right.attrs)
        ]
        if transforms[0] != transforms[1]:
            return False

    # 3/ Compare Dask graphs without computing arrays
    if is_dask_array(left.data) or is_dask_array(right.data):
        import_optional("dask")
        from dask.base import tokenize

        return tokenize(left.data) == tokenize(right.data)

    return left.equals(right)


def _rebuild_dataset(dataset: xr.Dataset, transformed: Mapping[str, xr.DataArray]) -> xr.Dataset:
    """
    Rebuild Xarray Dataset by replacing selected variables, and checking untouched variables against changed
    coordinates.

    The internal logic is the following:
    - _same_coordinate() identifies changed axes and georeferencing without computing lazy arrays.
    - Variables sharing those axes must be transformed together.
    - We then construct a Dataset from Variables, avoiding Xarray's automatic alignment of old values onto the
        new grid, and restore independent coordinates and Dataset metadata.
    """

    # 1/ First, we collect transformed coordinates and require every result to be on the same axes
    new_coordinates: dict[str, xr.Variable] = {}
    new_indexes: dict[str, xr.Index] = {}
    changed_dimensions: set[str] = set()
    changed_coordinates: set[str] = set()

    for name, result in transformed.items():
        # Record dimension sizes that changed
        original = dataset[name]
        for dimension in result.dims:
            if dimension in dataset.sizes and dataset.sizes[dimension] != result.sizes[dimension]:
                changed_dimensions.add(dimension)

        # Check coordinates carried by multiple variables for conflicting values or metadata
        for coordinate, value in result.coords.items():
            if coordinate in dataset.data_vars:
                # Point algorithms can carry an existing value variable as an auxiliary coordinate
                if not _same_coordinate(dataset[coordinate].variable, value.variable):
                    raise ValueError(f"Output coordinate {coordinate!r} conflicts with an existing value variable.")
                continue

            if coordinate in new_coordinates and not _same_coordinate(new_coordinates[coordinate], value.variable):
                raise ValueError(f"Transformed variables have conflicting coordinates for {coordinate!r}.")

            # Reuse original coordinates and indexes when unchanged, otherwise use transformed ones
            if coordinate in dataset.coords and _same_coordinate(dataset.coords[coordinate].variable, value.variable):
                new_coordinates[coordinate] = dataset.coords[coordinate].variable
                if coordinate in dataset.xindexes:
                    new_indexes[coordinate] = dataset.xindexes[coordinate]
            else:
                new_coordinates[coordinate] = value.variable
                if coordinate in result.xindexes:
                    new_indexes[coordinate] = result.xindexes[coordinate]

                # Coordinate changes affect every dimension they use
                if coordinate in dataset.coords:
                    changed_coordinates.add(coordinate)
                    changed_dimensions.update(value.dims)

        # CRS changes affect spatial meaning even when the destination uses identical numerical coordinates
        mapping = original.encoding.get("grid_mapping", original.attrs.get("grid_mapping"))
        if mapping in changed_coordinates:
            try:
                changed_dimensions.update((original.rio.x_dim, original.rio.y_dim))
            except MissingSpatialDimensionError:
                changed_dimensions.update(original.dims)

        # Compare CRS attributes for results with unchanged dimensions
        source_crs, target_crs = original.attrs.get("crs"), result.attrs.get("crs")
        if source_crs != target_crs and original.dims == result.dims:
            same_crs = (
                source_crs is not None
                and target_crs is not None
                and CRS.from_user_input(source_crs).equals(CRS.from_user_input(target_crs), ignore_axis_order=True)
            )
            if not same_crs:
                changed_dimensions.update(original.dims)

    # 2/ Refuse to attach untouched values to changed axes or a changed CRS
    conflicts = [
        name
        for name, value in dataset.data_vars.items()
        if name not in transformed
        and (
            changed_dimensions.intersection(value.dims)
            or value.encoding.get("grid_mapping", value.attrs.get("grid_mapping")) in changed_coordinates
        )
    ]
    if conflicts:
        raise ValueError(
            f"Untouched variables share transformed coordinates or dimensions: {conflicts}. "
            "Include them in 'variables' or separate them into another Dataset."
        )

    # 3/ Independent coordinates survive, dependent auxiliary coordinates need a transformed replacement
    coordinates = {}
    for name, coordinate in dataset.coords.items():
        if name in new_coordinates:
            continue
        if changed_dimensions.intersection(coordinate.dims):
            raise ValueError(f"Coordinate {name!r} shares transformed dimensions and has no transformed replacement.")
        coordinates[name] = coordinate.variable

    # We assemble variables explicitly to avoid automatic Xarray alignment onto the new coordinates
    coordinates.update(new_coordinates)
    values = {
        name: transformed[name].variable if name in transformed else value.variable
        for name, value in dataset.data_vars.items()
    }

    # 4/ Preserve existing indexes without creating a Pandas index from lazy point labels
    indexes = {name: index for name, index in dataset.xindexes.items() if name not in new_coordinates}
    indexes.update(new_indexes)

    # Construct Dataset and copy metadata and transformed variable encodings
    result = xr.Dataset(values, coords=xr.Coordinates(coordinates, indexes=indexes), attrs=dataset.attrs.copy())
    result.encoding = dataset.encoding.copy()
    for name, value in transformed.items():
        result[name].encoding = value.encoding.copy()

    return result
