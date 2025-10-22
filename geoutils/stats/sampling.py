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

"""Module for array sampling statistics."""

from __future__ import annotations

from typing import Literal, overload

import numpy as np

from geoutils._typing import MArrayNum, NDArrayNum
from geoutils.raster.array import get_mask_from_array


@overload
def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[False] = False,
    *,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum: ...


@overload
def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: Literal[True],
    *,
    random_state: int | np.random.Generator | None = None,
) -> tuple[NDArrayNum, ...]: ...


@overload
def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum | tuple[NDArrayNum, ...]: ...


def subsample_array(
    array: NDArrayNum | MArrayNum,
    subsample: float | int,
    return_indices: bool = False,
    random_state: int | np.random.Generator | None = None,
) -> NDArrayNum | tuple[NDArrayNum, ...]:
    """
    Randomly subsample a 1D or 2D array by a sampling factor, taking only non NaN/masked values.

    :param array: Input array.
    :param subsample: Subsample size. If <= 1, will be considered a fraction of valid pixels to extract.
        If > 1 will be considered the number of pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations (for testing)

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array)
    """
    # Define state for random sampling (to fix results during testing)
    rng = np.random.default_rng(random_state)

    # Remove invalid values and flatten array
    mask = get_mask_from_array(array)  # -> need to remove .squeeze in get_mask
    valids = np.argwhere(~mask.flatten()).squeeze()

    # Get number of points to extract
    # If subsample is one, we don't perform any subsampling operation, we return the valid array or indices directly
    if subsample == 1:
        unraveled_indices = np.unravel_index(valids, array.shape)
        if return_indices:
            return unraveled_indices
        else:
            return array[unraveled_indices]
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * np.count_nonzero(~mask))
    elif subsample > 1:
        npoints = int(subsample)
    else:
        raise ValueError("`subsample` must be > 0")

    # Checks that array and npoints are correct
    assert np.ndim(valids) == 1, "Something is wrong with array dimension, check input data and shape"
    if npoints > np.size(valids):
        npoints = np.size(valids)

    # Randomly extract npoints without replacement
    indices = rng.choice(valids, npoints, replace=False)
    unraveled_indices = np.unravel_index(indices, array.shape)

    if return_indices:
        return unraveled_indices
    else:
        return array[unraveled_indices]
