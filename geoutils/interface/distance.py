# Copyright (c) 2025 GeoUtils developers
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

"""Functionalities related to distance operations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt

from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.multiproc import MultiprocConfig, map_overlap
from geoutils.raster.referencing import _default_nodata

if TYPE_CHECKING:
    from geoutils.raster.base import RasterType
    from geoutils.raster.raster import Raster
    from geoutils.vector.vector import VectorType

try:
    import dask.array as da
except Exception:  # Keep Dask optional at import time
    da = None  # type: ignore


##############################
# 1/ INPUTS AND CONFIGURATION
##############################


def _validate_max_distance(max_distance: float | None) -> float | None:
    """Validate and normalize the optional max distance."""

    if max_distance is None:
        return None
    if isinstance(max_distance, (bool, np.bool_)) or not isinstance(
        max_distance, (int, float, np.integer, np.floating)
    ):
        raise TypeError("max_distance must be of type int or float.")

    max_distance = float(max_distance)
    if not np.isfinite(max_distance) or max_distance < 0:
        raise ValueError("max_distance must be non-negative and finite.")
    return max_distance


def _distance_sampling_and_overlap(
    raster: RasterType,
    distance_unit: Literal["pixel", "georeferenced"],
    max_distance: float | None,
) -> tuple[int | tuple[float | int, float | int], int]:
    """Return the SciPy sampling and overlap depth for the requested distance unit."""

    # Express distances with the same sampling used by the eager calculation
    if distance_unit.lower() == "georeferenced":
        sampling: int | tuple[float | int, float | int] = raster.res
        smallest_pixel_size = min(abs(float(resolution)) for resolution in raster.res)
    elif distance_unit.lower() == "pixel":
        sampling = 1
        smallest_pixel_size = 1
    else:
        raise ValueError('Distance unit must be either "georeferenced" or "pixel".')

    overlap = 0 if max_distance is None else math.ceil(max_distance / smallest_pixel_size)
    return sampling, overlap


def _vector_target_mask(
    raster: RasterType,
    vector: VectorType,
) -> Any:
    """Rasterize the vector current geometry as distance target cells on the raster grid."""

    # Mask pixels with vector geometry
    return vector.create_mask(raster, as_array=True)


def _raster_target_mask(raster: RasterType, target_values: list[float] | None) -> Any:
    """Define distance targets from an eager or Dask raster array."""

    # Get the raster array, converting masked values to NaNs while leaving a Dask source lazy
    raster_array = raster.data
    if np.ma.isMaskedArray(raster_array):
        if np.issubdtype(raster_array.dtype, np.integer):
            raster_array = raster_array.astype(np.float32).filled(np.nan)  # type: ignore[union-attr]
        else:
            raster_array = raster_array.filled(np.nan)

    # If input is a mask, target is implicit, and array needs to be converted to uint8
    if target_values is None and raster.is_mask:
        target_values = [1]
        raster_array = raster_array.astype("uint8")

    # Mask target pixels when values are provided
    if target_values is not None:
        if len(target_values) == 0:
            raise ValueError("target_values must contain at least one value.")
        target_mask = raster_array == target_values[0]
        for target_value in target_values[1:]:
            target_mask = np.logical_or(target_mask, raster_array == target_value)
        return target_mask

    # Otherwise, all non-zero values are considered targets
    return raster_array.astype(bool)


########################
# 2/ DISTANCE CALCULATION
########################


def _proximity_from_target_mask(
    target_mask: NDArrayBool,
    sampling: int | tuple[float | int, float | int],
    max_distance: float | None,
) -> NDArrayNum:
    """Calculate proximity for one complete array or overlapped array block."""

    # If there are no target pixels, pass an array full of nodata
    if np.count_nonzero(target_mask) == 0:
        return np.full(target_mask.shape, np.nan, dtype=np.float64)

    # For a multi-band raster, compute the distance matrix separately for each band
    if target_mask.ndim == 3:
        proximity = np.stack(
            [distance_transform_edt(~band_mask, sampling=sampling) for band_mask in target_mask], axis=0
        )
    else:
        proximity = distance_transform_edt(~target_mask, sampling=sampling)

    # Discard values whose nearest target may lie outside a chunk's finite overlap
    if max_distance is not None:
        proximity[proximity > max_distance] = np.nan
    return proximity


def _build_proximity_output(raster: RasterType, proximity: Any) -> Any:
    """Create a raster output with floating nodata and the source georeferencing."""

    return raster.from_array(
        data=proximity,
        transform=raster.transform,
        crs=raster.crs,
        nodata=_default_nodata(proximity.dtype),
        area_or_point=raster.area_or_point,
        tags=dict(raster.tags),
    )


#####################
# 3/ CHUNKED BACKENDS
#####################


def _dask_proximity(
    target_mask: Any,
    sampling: int | tuple[float | int, float | int],
    max_distance: float,
    overlap: int,
) -> Any:
    """Calculate proximity lazily with overlap on both spatial axes."""

    assert da is not None

    # Wrap in map_overlap with depth based on the max distance
    depth = (0,) * (target_mask.ndim - 2) + (overlap, overlap)
    return da.map_overlap(
        _proximity_from_target_mask,
        target_mask,
        depth=depth,
        boundary="none",
        dtype=np.float64,
        sampling=sampling,
        max_distance=max_distance,
    )


def _multiproc_proximity_block(
    block: Raster,
    vector: VectorType | None,
    target_values: list[float] | None,
    distance_unit: Literal["pixel", "georeferenced"],
    max_distance: float,
) -> Raster:
    """Calculate proximity on one padded raster block with multiprocessing."""

    # Prepare target cells on the padded block using the same raster or vector path as eager execution
    sampling, _ = _distance_sampling_and_overlap(block, distance_unit, max_distance)
    if vector is None:
        target_mask = _raster_target_mask(block, target_values)
    else:
        target_mask = _vector_target_mask(block, vector)

    # Calculate the padded result before map_overlap() crops it to the destination block
    proximity = _proximity_from_target_mask(target_mask, sampling=sampling, max_distance=max_distance)
    return _build_proximity_output(block, proximity)


####################
# 4/ BACKEND DISPATCH
####################


def _proximity_from_vector_or_raster(
    raster: RasterType,
    vector: VectorType | None = None,
    target_values: list[float] | None = None,
    distance_unit: Literal["pixel"] | Literal["georeferenced"] = "georeferenced",
    max_distance: float | None = None,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Calculate proximity to a Raster's target values if no Vector is provided, otherwise to a Vector's current geometry
    rasterized on the Raster, with eager, Dask, or multiprocessing execution.

    This function is defined here as mostly raster-based, but used in a class method for both Raster and Vector.

    Internally, it does the following:
    _raster_target_mask() or _vector_target_mask() first selects target cells.
    Eager and Dask execution then call _proximity_from_target_mask() directly, while multiprocessing applies the same
    steps to padded raster blocks in _multiproc_proximity_block().
    _build_proximity_output() restores the source grid and metadata for every backend.

    :param raster: Raster grid and optional raster values used to calculate proximity.
    :param vector: Vector whose current geometry is used as the target instead of raster values.
    :param target_values: Raster values used as targets. All nonzero values are targets by default.
    :param distance_unit: Calculate distance in georeferenced or pixel units.
    :param max_distance: Largest distance to return. Farther cells are set to nodata. Required for chunked execution.
    :param mp_config: Worker, chunk, and output settings for multiprocessing. Cannot be combined with Dask input.

    :returns: Proximity raster using the input raster grid and the selected execution backend.
    """

    # Validate user inputs
    max_distance = _validate_max_distance(max_distance)
    sampling, overlap = _distance_sampling_and_overlap(raster, distance_unit, max_distance)
    dask_backend = da is not None and raster._chunks is not None
    if mp_config is not None and dask_backend:
        raise ValueError("Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config.")
    if (dask_backend or mp_config is not None) and max_distance is None:
        raise ValueError("max_distance must be provided for Dask or multiprocessing proximity.")

    # With multiprocessing, rasterize targets and compute distances on padded blocks
    if mp_config is not None:
        if raster._is_xr:
            raise ValueError("Multiprocessing proximity requires a Raster input rather than an Xarray accessor.")
        assert max_distance is not None
        return map_overlap(
            _multiproc_proximity_block,
            raster,
            mp_config,
            vector,
            target_values,
            distance_unit,
            max_distance,
            depth=overlap,
        )

    # 1/ First, if there is a vector input, rasterize its current geometry
    if vector is not None:
        target_mask = _vector_target_mask(raster, vector)
    # Otherwise, mask target pixels from the raster values
    else:
        target_mask = _raster_target_mask(raster, target_values)

    # 2/ Now, compute the distance matrix relative to the masked vector geometry or raster target pixels
    if dask_backend:
        # Dask keeps the source chunks and adds the finite halo needed by the requested distance
        assert max_distance is not None
        proximity = _dask_proximity(
            target_mask,
            sampling=sampling,
            max_distance=max_distance,
            overlap=overlap,
        )
    else:
        proximity = _proximity_from_target_mask(target_mask, sampling=sampling, max_distance=max_distance)

    # 3/ Finally, construct the matching raster representation
    return _build_proximity_output(raster, proximity)
