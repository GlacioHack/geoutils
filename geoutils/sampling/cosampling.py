# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Sample several geospatial datasets at the same locations.

Note: This module is inspired from logic originally developed in xDEM for coregistration and uncertainty quantification.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import xarray as xr

from geoutils._dispatch import (
    _get_pointcloud_interface,
    _get_raster_interface,
    _is_raster,
    get_geo_attr,
    has_geo_attr,
    is_dask_array,
    is_dask_dataframe,
)
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, NDArrayBool, NDArrayNum
from geoutils.interface.gridding import GriddingMethod
from geoutils.raster.array import _selected_raster_data
from geoutils.sampling.subsampling import _sample_valid_indices
from geoutils.sampling.support import (
    _aligned_pointcloud,
    _aligned_raster,
    _as_array,
    _mask_at_support,
    _mask_on_raster,
    _normalize_sampling_input,
    _point_values_at_support,
    _sampling_specification,
    _sampling_support,
)

if TYPE_CHECKING:
    from geoutils.interface.interpolation import InterpolationMethod
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.raster.raster import Raster
    from geoutils.vector.base import VectorLike


#################################
# 1/ INPUT PREPARATION
#################################


@dataclass(frozen=True)
class _CosampleInput:
    """
    Small dataclass to consistently store pointers to normalized values, selector, input support and kind.

    These are created in the input preparation step done in _prepare_cosample_input(), right below!
    """

    value: RasterBase | PointCloudBase | ArrayLike
    selector: int | str | None
    input_support: RasterBase | PointCloudBase
    kind: Literal["raster", "point"]


def _prepare_cosample_input(
    value: Any,
    selector: int | str | None,
    name: str,
    native: _CosampleInput | None = None,
) -> _CosampleInput:
    """
    Prepare a single input: differentiate raster/pointclouds and arrays, validate their relative shape if they are an
    array input, and check their input band exists if provided.

    This function is used below in _prepare_cosample_all_inputs().
    """

    # Normalize Xarray inputs without spatial coordinates as arrays, and resolve spatial interfaces once
    value = _normalize_sampling_input(value)
    raster = _get_raster_interface(value)
    pointcloud = _get_pointcloud_interface(value) if raster is None else None
    kind: Literal["raster", "point"]
    if raster is not None:
        value = input_support = raster
        kind = "raster"
    elif pointcloud is not None:
        value = input_support = pointcloud
        kind = "point"
    else:
        # Require every array input to indicate the primary input support it follows
        if native is None:
            raise ValueError(f"Argument ``auxiliary_at`` must identify the native support of array auxiliary {name!r}.")
        input_support, kind = native.input_support, native.kind
        value = _as_array(value)

        # Check the native shape without loading Dask arrays or spatial values
        if kind == "raster":
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim != 2 or tuple(value.shape) != tuple(cast("RasterBase", input_support).shape):
                raise ValueError(f"Array {name!r} must match the shape of its native raster input.")
        elif value.ndim != 1:
            raise ValueError(f"Raw point value {name!r} must contain one value per native point.")

    # Validate raster bands from metadata before alignment, interpolation or worker dispatch
    if kind == "raster":
        if selector is None and name not in {"self", "other"}:
            selector = 1
        count = raster.count if raster is not None else 1
        if not isinstance(selector, (int, np.integer)):
            raise TypeError(f"Raster selector for {name!r} must be a band number.")
        if not 1 <= selector <= count:
            raise ValueError(f"Band for {name!r} must be between one and the raster band count.")
        selector = int(selector)
    elif pointcloud is not None:
        # Primary point inputs use their active values; auxiliary tuples can select another column
        selector = pointcloud.data_column if name in {"self", "other"} or selector is None else selector
        if selector is not None and (not isinstance(selector, str) or selector not in pointcloud.columns):
            raise ValueError(f"Point column {selector!r} selected for {name!r} does not exist.")
    else:
        selector = None

    return _CosampleInput(value, selector, input_support, kind)


def _prepare_cosample_inputs(
    first: RasterLike | PointCloudLike,
    second: RasterLike | PointCloudLike | ArrayLike,
    band: int,
    other_band: int,
    auxiliary: Mapping[str, Any] | None,
    auxiliary_at: Literal["self", "other"] | Mapping[str, Literal["self", "other"]] | None,
) -> dict[str, _CosampleInput]:
    """
    Prepare all inputs (storing kind, values, selector and native support) to later choose output support.

    See _cosample() for arguments.

    Raster/point cloud inputs use their own coordinates as input support. An array ``second`` follows the first
    input support, and array auxiliaries define their input support through ``auxiliary_at``.
    """

    # Copy auxiliary dictionaries to avoid modifying the user input, and check their content
    auxiliary = {} if auxiliary is None else dict(auxiliary)
    if any(not isinstance(name, str) or not name for name in auxiliary):
        raise ValueError("Auxiliary names must be non-empty strings.")
    if {"self", "other", "geometry"}.intersection(auxiliary):
        raise ValueError("Auxiliary names cannot be 'self', 'other' or 'geometry'.")
    if not isinstance(auxiliary_at, Mapping):
        if auxiliary_at not in (None, "self", "other"):
            raise ValueError("Values in argument ``auxiliary_at`` must be 'self' or 'other'.")
        auxiliary_at = {} if auxiliary_at is None else dict.fromkeys(auxiliary, auxiliary_at)
    else:
        auxiliary_at = dict(auxiliary_at)
    if not set(auxiliary_at).issubset(auxiliary):
        raise ValueError("Argument ``auxiliary_at`` contains a name that is not present in ``auxiliary``.")
    if any(location not in ("self", "other") for location in auxiliary_at.values()):
        raise ValueError("Values in argument ``auxiliary_at`` must be 'self' or 'other'.")

    # An array second input has to follow the first input's locations, including when explicitly selected as support
    inputs = {"self": _prepare_cosample_input(first, band, "self")}
    inputs["other"] = _prepare_cosample_input(second, other_band, "other", inputs["self"])

    # We loop through every auxiliary input
    for name, specification in auxiliary.items():
        # For a raster or point cloud, the input support is simply its coordinates
        # For arrays, we use the input specified for that auxiliary
        if isinstance(specification, tuple):
            value, selector = _sampling_specification(first, specification)
        else:
            value, selector = specification, None
        input_support_name = auxiliary_at.get(name)
        native = inputs[input_support_name] if input_support_name is not None else None
        inputs[name] = _prepare_cosample_input(value, selector, name, native)

    return inputs


def _check_cosample_input_types(
    inputs: Mapping[str, _CosampleInput],
    support: RasterBase | PointCloudBase,
    mask: RasterLike | VectorLike | ArrayLike | None,
) -> None:
    """
    Require inputs to be of the same object or accessor "family" (GeoUtils, or Xarray/Pandas).

    However, DataArrays and GeoDataFrames can mix eager and Dask. Plain arrays have to match their input locations,
    and vector outlines only supply a mask, so both of them are accepted in every case.
    """

    # Include all input supports
    values = [(name, input_data.input_support) for name, input_data in inputs.items()]
    values.append(("at", support))

    # On a grid, vector masks supply shapes rather than point values, including lazy geometry tables
    if _is_raster(mask) or not _is_raster(support):
        mask_interface = _get_raster_interface(mask)
        if mask_interface is None:
            mask_interface = _get_pointcloud_interface(mask)
        if mask_interface is not None:
            values.append(("mask", mask_interface))
    first = inputs["self"].input_support
    use_accessors = getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False)

    # Check the interface family (independently of whether its arrays or dataframe partitions are lazy)
    for name, interface in values:
        is_accessor = getattr(interface, "_is_xr", False) or getattr(interface, "_is_pd", False)
        if is_accessor != use_accessors:
            raise TypeError(
                f"Cannot mix Raster/PointCloud objects with DataArray/GeoDataFrame inputs in cosample(): {name!r}. "
                "Use one family for all geospatial inputs."
            )


def _choose_cosample_support(
    inputs: Mapping[str, _CosampleInput],
    at: Literal["self", "other"] | RasterLike | PointCloudLike | None,
    raster_point_mode: Literal["grid_points", "resample_raster"] | None,
) -> RasterBase | PointCloudBase:
    """
    Choose output locations based on the ``at`` input, and the raster-point mode.

    An ``at`` argument has precedence, otherwise the raster-point mode identifies the support.
    With neither option, _sampling_support() chooses the first point cloud, or the first raster's grid.

    The arguments fulfill different roles:
    - ``at`` can for instance decide which of 2 raster inputs is the reference, which raster_point_mode doesn't affect,
    - ``raster_point_mode`` decides on the direction in case of a raster-point comparison (grid points to raster,
      or resample raster at point coordinates), but can conflict with ``at`` if defined in the other direction.
    """

    # Choose explicit output locations before using the raster and point conversion direction
    if isinstance(at, str):
        if at not in {"self", "other"}:
            raise ValueError("Argument ``at`` must be 'self', 'other' or a geospatial support object.")
        at = inputs[at].input_support
    if at is None and raster_point_mode is not None:
        candidates = []
        kind = "raster" if raster_point_mode == "grid_points" else "point"
        for name in ("self", "other"):
            input_data = inputs[name]
            if input_data.kind == kind and input_data.value is input_data.input_support:
                candidates.append(input_data.input_support)
        if len(candidates) != 1:
            raise ValueError("The conversion mode requires one unambiguous input support; select ``at`` explicitly.")
        at = candidates[0]

    # Main call to select common support
    support = _sampling_support((inputs["self"].input_support, inputs["other"].input_support), at)

    # Reject a conversion direction that conflicts with the chosen output locations
    support_is_raster = _is_raster(support)
    if (support_is_raster and raster_point_mode == "resample_raster") or (
        not support_is_raster and raster_point_mode == "grid_points"
    ):
        raise ValueError(
            "Argument ``raster_point_mode`` conflicts with the grid or point locations selected by ``at``."
        )
    return support


########################################
# 2/ COMMON ALIGNMENT AND VALIDITY
########################################


def _align_cosample_inputs_for_raster_support(
    inputs: Mapping[str, _CosampleInput],
    support: RasterBase,
    grid_method: GriddingMethod,
    grid_kwargs: Mapping[str, Any],
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    temporary_files: ExitStack,
) -> tuple[dict[str, Any], dict[str, tuple[RasterBase, int]]]:
    """
    Align every input with the output support, keeping Dask/MP/eager support.

    Point inputs are gridded, and rasters reprojected to the support grid if necessary.
    NumPy and Dask use the selected arrays, while multiprocessing uses raster objects and band indexes that
    workers can read by tile.
    """

    from geoutils.pointcloud.dataframe import (
        _assign_point_values,
        _build_pointcloud_output,
        _get_dataframe_attrs,
    )

    # Store rasters and band numbers for multiprocessing, or store arrays for NumPy and Dask
    aligned_rasters = {}
    arrays = {}
    aligned_points = {}
    for name, input_data in inputs.items():
        value = input_data.value
        selected_band = cast(int, input_data.selector) if input_data.kind == "raster" else 1
        input_support = input_data.input_support

        # Give each gridded or reprojected input its own temporary file for multiprocessing
        intermediate = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None

        # Reuse each point projected coordinates for its spatial values and plain arrays
        if input_data.kind == "point":
            owner = cast("PointCloudBase", input_support)
            if id(owner) not in aligned_points:
                point_config = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
                aligned_points[id(owner)] = _aligned_pointcloud(owner, support, name, align, mp_config=point_config)
            pointcloud = aligned_points[id(owner)]
            if value is not owner:
                # Replace masked values with NaN and attach arrays by position, including single points and Dask chunks
                raw = value
                if np.ma.isMaskedArray(raw):
                    raw = np.where(np.ma.getmaskarray(raw), np.nan, np.ma.getdata(raw))
                geometry = pointcloud.ds[[pointcloud.ds.geometry.name]]
                if geometry.geometry.name != "geometry":
                    geometry = geometry.rename_geometry("geometry")
                dataframe = _assign_point_values(geometry, {name: raw})

                # Select the array column while keeping the owner's coordinates and spatial metadata
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Overriding 3D points with with data column", category=UserWarning
                    )
                    copied = _build_pointcloud_output(
                        dataframe,
                        data_column=name,
                        as_dataframe=pointcloud._is_pd,
                        attrs=_get_dataframe_attrs(pointcloud.ds),
                        preserve_locations=True,
                    )
                pointcloud = _get_pointcloud_interface(copied)

            # Select point columns without copying or loading the source, then calculate one raster band
            value = pointcloud.grid(
                ref=support,
                resampling=grid_method,
                data_column=cast("str | None", input_data.selector),
                mp_config=intermediate,
                **grid_kwargs,
            )

        # Align every input once, then keep its raster for workers or read its selected NumPy/Dask band
        raster = _aligned_raster(value, input_support, support, name, align, mp_config=intermediate)
        if mp_config is not None:
            aligned_rasters[name] = (raster, selected_band)
        else:
            arrays[name] = _selected_raster_data(raster, selected_band)

    return arrays, aligned_rasters


def _align_cosample_inputs_for_point_support(
    inputs: Mapping[str, _CosampleInput],
    support: PointCloudBase,
    partition_lengths: tuple[int, ...] | None,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    temporary_files: ExitStack,
) -> tuple[dict[str, Any], dict[str, tuple[RasterBase, int]]]:
    """
    Align every input with the output points and keep the representation needed for later sampling.

    Point inputs are checked lazily at the output support (as inputs need to be aligned already), while rasters are
    reprojected to the right CRS, and retained for later interpolation after the common validity mask is derived.
    """

    # Separate grid inputs from values already located at the output points
    point_values = {}
    grid_inputs = {}
    aligned_points = {}
    for name, input_data in inputs.items():
        if input_data.kind == "raster":
            grid_inputs[name] = input_data
            continue

        # Reject incompatible coordinates before any raster reprojection or interpolation starts
        owner = cast("PointCloudBase", input_data.input_support)
        if id(owner) not in aligned_points:
            intermediate = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
            aligned_points[id(owner)] = _aligned_pointcloud(owner, support, name, align, mp_config=intermediate)
        value = aligned_points[id(owner)] if input_data.value is owner else input_data.value
        point_values[name] = _point_values_at_support(
            cast("PointCloudBase | ArrayLike", value),
            input_data.selector,
            support_dataframe=support.ds,
            name=name,
            point_partition_lengths=partition_lengths,
        )

    # Give each raster reprojection its own temporary file and keep selected bands for deferred interpolation
    aligned_rasters = {}
    for name, input_data in grid_inputs.items():
        intermediate = temporary_files.enter_context(mp_config.temporary()) if mp_config is not None else None
        raster = _aligned_raster(
            input_data.value, input_data.input_support, support, name, align, mp_config=intermediate
        )
        aligned_rasters[name] = (raster, cast(int, input_data.selector))
    return point_values, aligned_rasters


def _intersect_validity(validity_layers: Iterable[Any]) -> Any:
    """Intersect boolean validity raster/pointcloud layers without computing lazy arrays."""

    # Combine finite coverage and mask eligibility while preserving the input array backend
    common_validity = None
    for validity in validity_layers:
        if validity is None:
            continue
        common_validity = validity if common_validity is None else common_validity & validity

    # Every cosampling path supplies validity from at least one primary input
    if common_validity is None:
        raise RuntimeError("Cosampling requires at least one validity layer.")
    return common_validity


###############################
# 3/ COSAMPLE ON RASTER SUPPORT
###############################


def _cosample_raster_eager(
    arrays: Mapping[str, NDArrayNum],
    common_validity: NDArrayBool,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
) -> NDArrayNum:
    """Cosample valid locations eagerly (in-memory) into a multi-band array output."""

    # Use all valid cells when no smaller sample was requested
    if subsample == 1:
        selected = common_validity
        if not np.any(common_validity):
            raise ValueError("There is no finite data common to all cosampled values.")
    else:
        # Subsample pixel locations randomly for all output bands
        rows, columns = _sample_valid_indices(
            common_validity, subsample=subsample, random_state=random_state, strategy=strategy
        )
        if rows.size == 0:
            raise ValueError("There is no finite data common to all cosampled values.")
        selected = np.zeros(common_validity.shape, dtype=bool)
        selected[rows, columns] = True

    # Stack self, other and auxiliary arrays as output bands, setting unselected cells to NaN
    return np.stack([np.where(selected, array, np.nan) for array in arrays.values()])


def _cosample_raster_dask(
    arrays: Mapping[str, Any],
    common_validity: Any,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
) -> Any:
    """
    Cosample valid locations lazily across chunks into a multi-band Dask array output.

    Same logic as eager, but written in Dask.
    """

    import_optional("dask")
    import dask.array as da

    # Use all valid cells when no smaller sample was requested
    if subsample == 1:
        selected = common_validity

        # Check that at least one cell is valid without computing the complete Dask output
        if not bool(common_validity.any().compute()):
            raise ValueError("There is no finite data common to all cosampled values.")
    else:
        # Randomly choose one set of cell positions for every output band
        rows, columns = _sample_valid_indices(
            common_validity, subsample=subsample, random_state=random_state, strategy=strategy
        )
        if rows.size == 0:
            raise ValueError("There is no finite data common to all cosampled values.")

        # Give each cell a unique number using row * width + column
        # Mark sampled cell numbers within each Dask chunk instead of loading the full mask into memory
        grid_rows = da.arange(common_validity.shape[0], chunks=common_validity.chunks[0])[:, None]
        grid_columns = da.arange(common_validity.shape[1], chunks=common_validity.chunks[1])[None, :]
        selected = da.isin(
            grid_rows * common_validity.shape[1] + grid_columns,
            rows * common_validity.shape[1] + columns,
        )

    # Stack self, other and auxiliary arrays as output bands, setting unselected cells to NaN
    return np.stack([np.where(selected, array, np.nan) for array in arrays.values()])


def _wrapper_cosample_raster_block_mp(
    tile: RasterBase,
    inputs: Mapping[str, tuple[RasterBase, int]],
    support: RasterBase,
    mask: RasterLike | VectorLike | ArrayLike | None,
    mask_mode: str,
    indices: tuple[NDArrayNum, NDArrayNum] | None = None,
    validity_only: bool = False,
) -> Raster:
    """
    Wrapper for Multiprocessing cosample in input blocks, used in _cosample_raster_mp() with map_overlap().

    Same logic as eager above, but for a chunk.
    """

    from geoutils.raster.raster import Raster

    # Read the same geographic window from every input and store its requested band in arrays
    tile = _get_raster_interface(tile)
    arrays = {}
    for name, (raster, band) in inputs.items():
        window = tile if name == "self" else raster.crop(tile.bounds)
        arrays[name] = _selected_raster_data(window, band)

    # Find this tile's first row and column in the complete output grid
    column, row = (~support.transform) * (tile.transform.c, tile.transform.f)
    row, column = int(round(row)), int(round(column))

    # Crop raster masks or slice array masks to this tile; vector masks are evaluated using coordinates
    if mask is not None:
        if _is_raster(mask):
            mask = get_geo_attr(mask, "crop", accessors=("rst",))(tile.bounds)
        elif not has_geo_attr(mask, "create_mask", accessors=("vct",)):
            mask = _as_array(mask).reshape(support.shape)[row : row + tile.height, column : column + tile.width]

    # Intersect the user mask and finite coverage from every input before selecting any cells
    validity_layers = [_mask_at_support(mask, tile, mask_mode=mask_mode)]
    validity_layers.extend(np.isfinite(array) for array in arrays.values())
    common_validity = _intersect_validity(validity_layers)

    # Find sampled cells within this tile's rows, then check that their columns also fall inside the tile
    # The sample is sorted by row so each tile can look up its cells without scanning the full sample
    if indices is not None:
        lower, upper = np.searchsorted(indices[0], (row, row + tile.height))
        rows, columns = indices[0][lower:upper] - row, indices[1][lower:upper] - column
        inside = (columns >= 0) & (columns < tile.width)
        selected = np.zeros(tile.shape, dtype=bool)
        selected[rows[inside], columns[inside]] = True
        common_validity &= selected

    # Return a band of 1 for valid cells and NaN elsewhere when choosing the sample
    # For the final output, return one band per input with excluded cells set to NaN
    if validity_only:
        data = np.where(common_validity, np.float32(1), np.float32(np.nan))
    else:
        data = np.stack([np.where(common_validity, array, np.nan) for array in arrays.values()])
    return Raster.from_array(
        data,
        tile.transform,
        tile.crs,
        nodata=np.nan,
        area_or_point=support.area_or_point,
        tags={} if validity_only else {"long_name": tuple(inputs)},
    )


def _wrapper_has_finite_raster_block(tile: RasterBase) -> bool:
    """
    Wrapper for Multiprocessing through map_blocks: check one validity tile for finite cells."""

    return bool(np.any(np.isfinite(_selected_raster_data(tile))))


def _cosample_raster_mp(
    inputs: Mapping[str, tuple[RasterBase, int]],
    support: RasterBase,
    mask: RasterLike | VectorLike | ArrayLike | None,
    mask_mode: str,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    mp_config: MultiprocConfig,
    temporary_files: ExitStack,
) -> Raster:
    """
    Cosample valid values from a temporary "validity" file, then write their values by block.

    Same logic as eager above, but chunked using Multiprocessing as backend.

    _wrapper_cosample_raster_block_mp() applies the same mask in both passes.
    We keep the temporary "validity" file alive until the values have been written, and sort
    sampled rows so each worker can find the cells inside its block.
    """

    from geoutils.multiproc import map_blocks, map_overlap

    # Write 1 where all inputs are finite and the mask allows the cell, and NaN elsewhere
    # Select cells from this temporary file before writing the output values
    intermediate = temporary_files.enter_context(mp_config.temporary())
    reference = inputs["self"][0]
    validity_raster = map_overlap(
        _wrapper_cosample_raster_block_mp, reference, intermediate, inputs, support, mask, mask_mode, validity_only=True
    )

    # Check that valid cells exist, or randomly select a subset for all output bands
    indices = None
    if subsample == 1:
        has_valid = any(map_blocks(_wrapper_has_finite_raster_block, validity_raster, intermediate))
    else:
        indices = validity_raster.subsample(
            subsample, return_indices=True, random_state=random_state, strategy=strategy, mp_config=intermediate
        )
        has_valid = len(indices[0]) > 0

        # Sort sampled cells by row so each tile can find its cells without scanning the full sample
        order = np.argsort(indices[0], kind="stable")
        indices = indices[0][order], indices[1][order]
    if not has_valid:
        raise ValueError("There is no finite data common to all cosampled values.")

    # Read each tile again to write the selected values from all inputs to the final output file
    return map_overlap(
        _wrapper_cosample_raster_block_mp, reference, mp_config, inputs, support, mask, mask_mode, indices=indices
    )


def _cosample_on_raster(
    first: RasterLike | PointCloudLike,
    inputs: Mapping[str, _CosampleInput],
    *,
    support: RasterBase,
    mask: RasterLike | VectorLike | ArrayLike | None,
    mask_mode: str,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    grid_method: GriddingMethod,
    grid_kwargs: Mapping[str, Any],
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    temporary_files: ExitStack,
) -> RasterLike:
    """
    Cosample all inputs at the same raster locations into a multi-band raster output.

    See _cosample() for most input arguments.

    This function does in order:

    - _align_cosample_inputs_for_raster_support() places every input on the grid.
    - _intersect_validity() then combines finite coverage with the optional mask into a common validity mask.
    - Then an eager/Dask/MP _cosample_raster() function draws one common sample and constructs the multi-band output.

    :param inputs: Named values with their selected band and original locations, prepared by _cosample().
    :param support: Raster defining the output grid, resolved from ``at`` and ``raster_point_mode``.
    :param temporary_files: Pass ExitStack to keep intermediate multiprocessing files alive until output is built.

    :returns: Raster with one band per input and a shared mask outside the selected cells.
    """

    # 1/ Align every input with the output grid without loading lazy values
    arrays, aligned_rasters = _align_cosample_inputs_for_raster_support(
        inputs, support, grid_method, grid_kwargs, align, mp_config, temporary_files
    )

    # 2/ Compute the common validity mask, then select the same sample of grid cells for all aligned inputs

    # 2a/ For Multiprocessing
    if mp_config is not None:
        # Align raster masks once before workers crop them, and validate array masks against the complete grid
        if _is_raster(mask):
            intermediate = temporary_files.enter_context(mp_config.temporary())
            mask = _aligned_raster(mask, mask, support, "mask", align, mp_config=intermediate)
        elif mask is not None and not has_geo_attr(mask, "create_mask", accessors=("vct",)):
            mask = _mask_on_raster(mask, support, mask_mode, align)

        result = _cosample_raster_mp(
            aligned_rasters,
            support,
            mask,
            mask_mode,
            subsample,
            random_state,
            strategy,
            mp_config,
            temporary_files,
        )

        # Return the output file through the caller's raster interface
        output = first._cast_raster_output(result)
        if isinstance(output, xr.DataArray):
            output.attrs["long_name"] = tuple(inputs)
        return output

    # 2b/ For Dask or eager, intersect the user mask and finite coverage from every aligned value
    validity_layers = [_mask_at_support(mask, support, mask_mode=mask_mode, align=align)]
    validity_layers.extend(np.isfinite(array) for array in arrays.values())
    common_validity = _intersect_validity(validity_layers)

    # Subsample only the common valid cells, then stack their values into output bands
    if is_dask_array(common_validity):
        data = _cosample_raster_dask(arrays, common_validity, subsample, random_state, strategy)
    else:
        data = _cosample_raster_eager(arrays, common_validity, subsample, random_state, strategy)

    # 3/ Construct the raster output from the NumPy or Dask bands
    tags = {"long_name": tuple(arrays)}

    # Return an Xarray for accessor calls without computing its Dask arrays
    if getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False):
        from geoutils.raster.xr_accessor import RasterAccessor

        return RasterAccessor.from_array(
            data, support.transform, support.crs, nodata=np.nan, area_or_point=support.area_or_point, tags=tags
        )

    # Load the result into a Raster
    from geoutils.raster.raster import Raster

    data = data.compute() if is_dask_array(data) else data
    return Raster.from_array(
        data, support.transform, support.crs, nodata=np.nan, area_or_point=support.area_or_point, tags=tags
    )


##############################
# 4/ COSAMPLE ON POINT SUPPORT
##############################


def _raster_valid_at_points(
    raster: RasterBase,
    points: PointCloudLike | tuple[NDArrayNum, NDArrayNum],
    resample_method: InterpolationMethod,
    band: int,
    resample_kwargs: Mapping[str, Any],
    mp_config: MultiprocConfig | None = None,
    point_partition_lengths: tuple[int, ...] | None = None,
) -> Any:
    """
    Find valid raster points by interpolating raster block's validity, respecting Dask/MP.

    Arguments follow _cosample_on_points(), except for point_partition_lengths that follows _point_values_at_support().
    """

    # Interpolate a mask of 1 for finite raster cells and NaN for missing cells to find points with data
    # Use no extra nodata spreading unless the caller requests it
    values = raster.interp_points(
        points=points,
        method=resample_method,
        band=band,
        as_array=not is_dask_dataframe(points),
        mp_config=mp_config,
        _validity_only=True,
        **{"dist_nodata_spread": 0, **resample_kwargs},
    )

    # Convert the interpolated Dask column to an array, reusing point counts per chunk when available
    if is_dask_dataframe(values):
        values = get_geo_attr(values, "data", ("pc",)).to_dask_array(lengths=point_partition_lengths)
    return np.isfinite(values)


def _cosample_on_points(
    first: RasterLike | PointCloudLike,
    inputs: Mapping[str, _CosampleInput],
    *,
    support: PointCloudBase,
    mask: RasterLike | VectorLike | ArrayLike | None,
    mask_mode: str,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    resample_method: InterpolationMethod,
    resample_kwargs: Mapping[str, Any],
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None,
    temporary_files: ExitStack,
) -> PointCloudLike:
    """
    Cosample all inputs at the same point locations into a multi-column point cloud output.

    See _cosample() for input arguments.

    This function does in order:
    - _align_cosample_inputs_for_point_support() checks point locations and separates their values from aligned rasters,
    - _raster_valid_at_points() checks raster valid values before defining a common validity mask,
    - Finally, raster values are interpolated at the selected points with interp_points().

    :param support: Point cloud defining the ordered output coordinates, resolved from ``at`` and ``raster_point_mode``.
    :param inputs: Named values, raster bands and original locations resolved by _prepare_cosample_inputs().
    :param temporary_files: Pass ExitStack to keep intermediate multiprocessing files alive until output is built.

    :returns: PointCloud with one column per input, retaining selected points in their original order.
    """

    from geoutils.pointcloud.dataframe import (
        _assign_point_values,
        _build_pointcloud_output,
        _point_partition_lengths,
        _select_point_rows,
    )

    # 1/ Read the output point coordinates and align each input for sampling
    dataframe = support.ds
    is_dask_points = is_dask_dataframe(dataframe)

    # Count rows in each Dask chunk so values and masks can be matched to the same points
    # Keep the Dask dataframe to not load all coordinates in memory
    partition_lengths = _point_partition_lengths(dataframe) if is_dask_points else None

    # Check every point input before aligning rasters, whose values will be read only at the selected points
    point_values, aligned_rasters = _align_cosample_inputs_for_point_support(
        inputs, support, partition_lengths, align, mp_config, temporary_files
    )

    # 2/ Compute the common validity mask, then select the same sample of point locations for all aligned inputs
    validity_layers = [np.isfinite(values) for values in point_values.values()]

    # Check where each raster has data and combine the result with the masks from the point values
    if aligned_rasters:
        points = dataframe if is_dask_points else (dataframe.geometry.x.to_numpy(), dataframe.geometry.y.to_numpy())
    for raster, selected_band in aligned_rasters.values():
        finite = _raster_valid_at_points(
            raster, points, resample_method, selected_band, resample_kwargs, mp_config, partition_lengths
        )
        validity_layers.append(finite)

    # Read the user mask at the output point coordinates and exclude points where it is False
    intermediate = (
        temporary_files.enter_context(mp_config.temporary()) if mp_config is not None and mask is not None else None
    )
    mask_values = _mask_at_support(
        mask,
        support,
        support_dataframe=dataframe,
        mask_mode=mask_mode,
        align=align,
        mp_config=intermediate,
        point_partition_lengths=partition_lengths,
    )
    validity_layers.append(mask_values)

    # Intersect all finite coverage and mask eligibility before drawing one common sample
    common_validity = _intersect_validity(validity_layers)

    # Keep all remaining points, or randomly choose a smaller sample as requested
    selected_rows = common_validity
    if subsample != 1:
        (indices,) = _sample_valid_indices(
            common_validity, subsample=subsample, random_state=random_state, strategy="sequential"
        )
        if indices.size == 0:
            raise ValueError("There is no finite data common to all cosampled values.")

        # Sort the selected row numbers so the output follows the original point order
        selected_rows = np.sort(indices)
    elif is_dask_points and not bool(common_validity.any().compute()):
        raise ValueError("There is no finite data common to all cosampled values.")

    # Create a table with only geometry and the requested point columns, then keep the selected rows
    geometry = dataframe[[dataframe.geometry.name]]
    if geometry.geometry.name != "geometry":
        geometry = geometry.rename_geometry("geometry")
    point_columns = _assign_point_values(geometry, point_values, partition_lengths=partition_lengths)
    output = _select_point_rows(point_columns, selected_rows, partition_lengths=partition_lengths)
    if not is_dask_dataframe(output) and output.empty:
        raise ValueError("There is no finite data common to all cosampled values.")

    # 3/ Interpolate each raster at the selected coordinates and add its values as a new column
    if aligned_rasters:
        selected_points = (
            output if is_dask_dataframe(output) else (output.geometry.x.to_numpy(), output.geometry.y.to_numpy())
        )
        sampled = {}
        for name, (raster, selected_band) in aligned_rasters.items():
            values = raster.interp_points(
                points=selected_points,
                method=resample_method,
                band=selected_band,
                as_array=not is_dask_dataframe(output),
                mp_config=mp_config,
                **resample_kwargs,
            )

            # Extract the interpolated column as a Dask Series so it keeps the same chunks as the selected points
            sampled[name] = get_geo_attr(values, "data", ("pc",)) if is_dask_dataframe(values) else values

        # Final interpolation may exclude more points near nodata; remove rows with missing or infinite values
        output = _assign_point_values(output, sampled)
        output = output.replace([np.inf, -np.inf], np.nan).dropna(subset=list(inputs))

    # Order all columns as self, other, auxiliaries, then geometry
    output = output[[*inputs, "geometry"]]

    # Build the point output with self as its active column and metadata for the selected rows
    # Accessor calls keep Dask data chunked; PointCloud calls load the result into memory
    as_dataframe = getattr(first, "_is_xr", False) or getattr(first, "_is_pd", False)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Overriding 3D points with with data column 'self'", category=UserWarning
        )
        result = _build_pointcloud_output(output, data_column="self", as_dataframe=as_dataframe)

    # Check for empty results only when loaded, so Dask does not run the final interpolation yet
    if not is_dask_dataframe(result) and get_geo_attr(result, "point_count", ("pc",)) == 0:
        raise ValueError("There is no finite data common to all cosampled values.")
    return result


###########################
# 5/ MAIN COSAMPLE FUNCTION
###########################


def _cosample(
    first: RasterLike | PointCloudLike,
    second: RasterLike | PointCloudLike | ArrayLike,
    *,
    band: int,
    other_band: int,
    auxiliary: Mapping[str, Any] | None,
    auxiliary_at: Literal["self", "other"] | Mapping[str, Literal["self", "other"]] | None,
    at: Literal["self", "other"] | RasterLike | PointCloudLike | None,
    mask: RasterLike | VectorLike | ArrayLike | None,
    mask_mode: Literal["inside", "outside"],
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["sequential", "topk"],
    raster_point_mode: Literal["grid_points", "resample_raster"] | None,
    grid_method: GriddingMethod,
    resample_method: InterpolationMethod | Literal["reduce"],
    grid_kwargs: Mapping[str, Any] | None,
    resample_kwargs: Mapping[str, Any] | None,
    align: Literal["raise", "reproject"],
    mp_config: MultiprocConfig | None = None,
) -> RasterLike | PointCloudLike:
    """
    Cosample two datasets at the same locations, potentially with same-shape auxiliary data tied to them.

    Two primary inputs can be rasters or point clouds, and auxiliary outputs can also be arrays but need to match the
    shape of one of the two main inputs.
    The output is a multi-band raster or multi-column point cloud containing all data sampled at valid values of the
    same locations.

    This function reuses eager, Dask and multiprocessing execution from other geospatial operations (grid,
    interp_points). Additional steps include placing values on common support, determine validity (non-NaN/inf),
    then drawing one shared sample.
    Specifically:
    - Dask retains array chunks or point partitions in a graph.
    - Multiprocessing uses temporary raster files and map_overlap() to write the final bands without holding chunks
    in-memory at once.

    This main function has the following steps:

    - First, _prepare_cosample_inputs() resolves each input's values, band and original locations.
    - Then _choose_cosample_support() identifies the shared output locations and _check_cosample_input_types()
      enforces consistent object types (GeoUtils or Xarray/Pandas).
    - Finally, _cosample_on_raster() or _cosample_on_points() aligns the inputs, computes a common validity mask, then
      selects inputs at the same location (with optional subsampling), and finally builds the output.

    :param first: First raster or point cloud, whose selected values become the "self" output.
    :param second: Second raster or point cloud to sample alongside the first. An array has to match the first input
        grid shape or point count.
    :param band: Band selected from the first raster, counting from one. Point clouds use their main data column.
    :param other_band: Band selected from the second input if it is a raster, counting from one.
    :param auxiliary: Additional values by output name (e.g. {"slope": slope_raster}). Select a raster band or
        point column with a pair, e.g. {"slope": (slope_raster, 2)} or {"intensity": (points, "intensity")}.
        Spatial inputs default to the first raster band or active point values.
    :param auxiliary_at: Input locations followed by plain auxiliary arrays: "self", "other", or a choice per name
        (e.g. {"slope": "other"}). Spatial auxiliaries use their own coordinates.
    :param at: Output locations: "self", "other", or a reference raster/point cloud. Defaults to the first point
        cloud, or first raster's grid when neither input is a point cloud.
    :param mask: Locations eligible for sampling, defined by a boolean array, spatial mask, or vector outlines.
    :param mask_mode: Whether a vector mask keeps locations "inside" or "outside" its geometries.
    :param subsample: Fraction of common finite locations (e.g. 0.1), or maximum count (e.g. 1000); 1 keeps all.
    :param random_state: Seed or random generator for reproducible sampling (e.g. 42).
    :param strategy: Raster sampling with "topk" or "sequential"; "topk" keeps the same seeded sample across chunk
        sizes. Point output always uses "sequential".
    :param raster_point_mode: Conversion direction: "grid_points" places points on a raster, "resample_raster" reads
        rasters at points. Defaults to at's locations, or point locations when available. Must agree with at.
    :param grid_method: Point gridding by SciPy interpolation ("nearest", "linear", "cubic"), or circular "idw",
        "mean", "minimum", "maximum", "range", "count", "stdev", "average_distance", "average_distance_pts".
        The aliases "average", "min" and "max" select "mean", "minimum" and "maximum".
    :param resample_method: Raster interpolation using the SciPy methods "nearest", "linear", "cubic", "quintic",
        "slinear", "pchip" or "splinef2d". Window reduction ("reduce") is not implemented.
    :param grid_kwargs: Options for PointCloud.grid(), e.g. {"dist_nodata_pixel": 2, "min_points": 3} sets a two-pixel
        radius and minimum of three finite points for circular methods. Other options include "distance_power" for
        IDW and "engine" ("scipy" or "numba"). Set output locations and method with at and grid_method.
    :param resample_kwargs: Options for Raster.interp_points(), e.g. {"nodata_propagation": "ignore"}. The nodata
        policies are "gdal", "ignore" and "propagate"; "dist_nodata_spread" controls extra spreading in pixels.
        Set locations, band and method with the corresponding cosample() arguments.
    :param align: Handling of mismatched grids or coordinate systems: "raise" an error, or "reproject" to match at.
        Point inputs must still share the same ordered coordinates when sampled at points.
    :param mp_config: Worker and tile settings for multiprocessing. Raster output uses its outfile; cannot be
        combined with Dask inputs.

    :returns: Raster bands or point cloud columns named 'self', 'other' and the auxiliaries.
        All values share the same finite locations; raster cells outside the sample remain masked.
    """

    # 1/ Check input arguments and raise appropriate errors
    # Basic type/value checks
    if second is None:
        raise TypeError("Argument ``other`` is required for cosample().")
    if mask_mode not in {"inside", "outside"}:
        raise ValueError("Argument ``mask_mode`` must be 'inside' or 'outside'.")
    if strategy not in {"sequential", "topk"}:
        raise ValueError("Argument ``strategy`` must be 'sequential' or 'topk'.")
    if align not in {"raise", "reproject"}:
        raise ValueError("Argument ``align`` must be 'raise' or 'reproject'.")
    if not isinstance(subsample, (int, float)) or subsample <= 0:
        raise ValueError("Argument ``subsample`` must be a positive number.")
    if raster_point_mode not in {None, "grid_points", "resample_raster"}:
        raise ValueError("Argument ``raster_point_mode`` must be 'grid_points', 'resample_raster' or None.")

    # Copy extra options, and reject arguments that should be passed to cosample() directly and not as kwargs
    # (This check is required because interp_points() contains similarly-named inputs as cosample(), such as
    # "mp_config" or "band", etc)
    grid_kwargs = {} if grid_kwargs is None else dict(grid_kwargs)
    resample_kwargs = {} if resample_kwargs is None else dict(resample_kwargs)
    if {"ref", "grid_coords", "res", "shape", "bounds", "resampling", "data_column", "mp_config"}.intersection(
        grid_kwargs
    ):
        raise ValueError(
            "Use ``at``, ``grid_method``, point column selectors and ``mp_config`` outside ``grid_kwargs`` "
            "to choose the grid, method, point column and backend."
        )
    if {
        "points",
        "method",
        "band",
        "as_array",
        "input_latlon",
        "return_interpolator",
        "mp_config",
        "_validity_only",
    }.intersection(resample_kwargs):
        raise ValueError(
            "Use ``at``, ``band``, ``resample_method`` and ``mp_config`` outside ``resample_kwargs``; "
            "validity is managed internally."
        )

    # 2/ Resolve input locations and define the shared output support
    inputs = _prepare_cosample_inputs(first, second, band, other_band, auxiliary, auxiliary_at)
    mask = _normalize_sampling_input(mask)
    support = _choose_cosample_support(inputs, at, raster_point_mode)
    _check_cosample_input_types(inputs, support, mask)

    # Find the raster or point cloud interface that defines the output locations
    support_is_raster = _is_raster(support)
    if not support_is_raster and resample_method == "reduce":
        raise NotImplementedError(
            "Window reduction in cosample awaits revision of Raster.reduce_points(); "
            "use reduce_points separately in the meantime."
        )

    # 3/ Select execution backend and, for Multiprocessing, keep temporary files alive until the output is complete

    # If any input is Dask but mp_config was passed, raise an error
    if mp_config is not None:
        input_values = [input_data.value for input_data in inputs.values()]
        for value in [*input_values, support, mask]:
            # Inspect raw collections and spatial metadata without reading file-backed DataArray values
            lazy = is_dask_array(value) or is_dask_dataframe(value)
            if has_geo_attr(value, "_chunks", accessors=("rst",)):
                lazy |= get_geo_attr(value, "_chunks", accessors=("rst",)) is not None
            if has_geo_attr(value, "_is_dask", accessors=("pc", "vct")):
                lazy |= get_geo_attr(value, "_is_dask", accessors=("pc", "vct"))
            if lazy:
                raise ValueError("Cannot use Multiprocessing and Dask simultaneously in cosample().")

    # Keep temporary files alive until end of execution with ExitStack()
    with ExitStack() as temporary_files:
        # Dispatch only the kwargs relevant to the support operation (grid() or interp_points())
        if support_is_raster:
            return _cosample_on_raster(
                first,
                inputs,
                support=cast("RasterBase", support),
                mask=mask,
                mask_mode=mask_mode,
                subsample=subsample,
                random_state=random_state,
                strategy=strategy,
                grid_method=grid_method,
                grid_kwargs=grid_kwargs,
                align=align,
                mp_config=mp_config,
                temporary_files=temporary_files,
            )

        return _cosample_on_points(
            first,
            inputs,
            support=cast("PointCloudBase", support),
            mask=mask,
            mask_mode=mask_mode,
            subsample=subsample,
            random_state=random_state,
            resample_method=cast("InterpolationMethod", resample_method),
            resample_kwargs=resample_kwargs,
            align=align,
            mp_config=mp_config,
            temporary_files=temporary_files,
        )
