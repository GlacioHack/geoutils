# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
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

"""
This module allows the user to process grouped statistics easily
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from geoutils._typing import NDArrayNum
from geoutils.raster.raster import Raster, RasterType


def from_raster_to_flattened(dict_raster: dict[str, Any]) -> dict[str, Any]:
    """
    Transform every raster in flattened data to be processed by pandas
    :param: Dictionary containing binning name associated to be grouped by and a raster
    """
    dict_arrays = dict_raster.copy()

    for key, raster in dict_raster.items():
        dict_arrays[key] = raster.data.data.flatten()

    return dict_arrays


def is_interval(bin_value: RasterType | list[int] | object) -> bool:
    """
    Verifying the interval coherence if binning is a list
    :param bin_value: Bin value
    """
    if isinstance(bin_value, list):
        bins_array = np.asarray(bin_value)
    else:
        return False

    if bins_array.ndim != 1:
        raise ValueError("If bins is an interval, it must be a 1-dimensional array")
    elif not np.issubdtype(bins_array.dtype, np.number):
        raise TypeError("Bins must be a list of number")
    elif bins_array.size < 2:
        raise ValueError("Bins must be of size >= 2")
    elif not np.all(np.diff(bins_array) > 0):
        raise ValueError("Values must be strictly increasing.")

    return True


def grouped_stats(
    groupby_vars: dict[str, RasterType],
    bins: dict[str, RasterType | list[int] | object],
    aggregated_vars: dict[str, NDArrayNum],
    statistics: list[str],
) -> tuple[Any, dict[str, dict[str, NDArrayNum] | None]]:
    """
    Get statistics grouped (=binned) by other variables, whether categorical or continuous.
    :param groupby_vars: Dictionary containing Raster to group by.
    :param bins: Bins to use. Can be a list of interval, binary mask or segmentation map.
    :param aggregated_vars: Values to group, can be a dictionary .
    :param statistics: List or dict of statistics to compute, e.g. ["mean", "std"].
    """

    # Check that groupby_vars and bins as the same values
    if set(groupby_vars) != set(bins):
        raise ValueError("One bins/mask/segmentation entry per input array required.")

    # Load bins
    new_bins = {}
    for key, value in bins.items():
        if isinstance(value, str):
            new_bins[key] = Raster(value)
            logging.info(f"No need to save {value} again")
        else:
            new_bins[key] = value

    raster_base = next(iter(groupby_vars.values()))
    crs = raster_base.crs
    shape = raster_base.data.shape
    transform = raster_base.transform
    nodata = raster_base.nodata

    # Flatten arrays for pandas
    groupby_vars_array = from_raster_to_flattened(groupby_vars)
    aggregated_vars_arrays = from_raster_to_flattened(aggregated_vars)

    df = pd.DataFrame({**groupby_vars_array, **aggregated_vars_arrays})
    if nodata is not None:
        df = df.replace(nodata, np.nan)

    groupby_keys = []
    returned_masks = {}

    for groupby_key in groupby_vars.keys():
        group_col = f"groupby_{groupby_key}"
        bins_array = new_bins[groupby_key]

        if isinstance(bins[groupby_key], (str, Raster)):
            returned_masks[group_col] = None
        else:
            returned_masks[group_col] = {}

        # ---------- Interval bins ----------
        if is_interval(bins_array):
            cut = pd.cut(df[groupby_key], bins=bins_array)
            df[group_col] = cut

            if returned_masks[group_col] is not None:
                for interval in cut.cat.categories:
                    mask_flat = (cut == interval).to_numpy()
                    mask_to_raster = Raster.from_array(mask_flat.reshape(shape), transform, crs)

                    returned_masks[group_col][f"{groupby_key}_{interval.left}_{interval.right}"] = mask_to_raster

        # ---------- Raster (mask or segmentation) ----------
        elif isinstance(bins_array, Raster):
            data = np.asarray(bins_array.data.data).flatten()

            if len(data) != len(df):
                raise ValueError(f"Mask has invalid length {len(data)}, expected {len(df)}")

            # Binary mask
            if bins_array.is_mask:
                df[group_col] = pd.Categorical(data, categories=[False, True])

                if returned_masks[group_col] is not None:
                    mask_to_raster = Raster.from_array(data.reshape(shape).astype("uint8"), transform, crs)

                    returned_masks[group_col][f"{groupby_key}_mask"] = mask_to_raster

            # Segmentation
            else:
                df[group_col] = pd.Categorical(data)

                if returned_masks[group_col] is not None:
                    segm_names = np.unique(data[~np.isnan(data)])
                    for segm_name in segm_names:
                        mask_to_raster = Raster.from_array((data == segm_name).reshape(shape), transform, crs)

                        returned_masks[group_col][f"{groupby_key}_seg_{int(segm_name)}"] = mask_to_raster

        else:
            raise NotImplementedError("This type of bins does not yet work")

        groupby_keys.append(group_col)

    # Compute statistics
    result = df.groupby(groupby_keys, observed=True)[list(aggregated_vars.keys())].agg(statistics)

    return result, returned_masks
