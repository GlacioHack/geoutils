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

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geoutils._typing import NDArrayNum
from geoutils.raster.raster import Raster, RasterType


def init_binnings_attributes(raster: RasterType) -> tuple[Any, Any, Any]:
    """
    Initialize binning attributes for saving tiff file.

    :param raster: raster object
    """
    crs = raster.crs
    shape = raster.data.shape
    transform = raster.transform

    return crs, shape, transform


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
    elif bins_array.size <= 2:
        raise ValueError("Bins must be of size > 2")
    elif not np.all(np.diff(bins_array) > 0):
        raise ValueError("Values must be strictly increasing.")

    return True


def grouped_stats(
    groupby_vars: dict[str, RasterType],
    bins: dict[str, RasterType | list[int] | object],
    aggregated_vars: dict[str, NDArrayNum],
    statistics: list[str],
    save_csv: str | Path | None = None,
    save_masks: str | Path | None = None,
) -> pd.DataFrame:
    """
    Get statistics grouped (=binned) by other variables, whether categorical or continuous.
    :param groupby_vars: Dictionary containing Raster to group by.
    :param bins: Bins to use. Can be a list of interval, binary mask or segmentation map.
    :param aggregated_vars: Values to group, can be a dictionary .
    :param statistics: List or dict of statistics to compute, e.g. ["mean", "std"].
    :param save_csv: Path to save CSV file.
    :param save_masks: Path to save masks.
    """

    # Check that groupby_vars and bins as the same values
    if set(groupby_vars) != set(bins):
        raise ValueError("One bins/mask/segmentation entry per input array required.")

    crs, shape, transform = init_binnings_attributes(next(iter(groupby_vars.values())))

    # panda dataframe works on arrays
    groupby_vars_array = from_raster_to_flattened(groupby_vars)
    aggregated_vars_arrays = from_raster_to_flattened(aggregated_vars)

    # Init panda data frame
    df = pd.DataFrame({**groupby_vars_array, **aggregated_vars_arrays})
    groupby_keys = []
    # Analyze type of binnings
    for groupby_key in groupby_vars.keys():
        group_col = f"groupby_{groupby_key}"

        bins_array = bins[groupby_key]

        # is interval
        if is_interval(bins_array):
            df[group_col] = pd.cut(df[groupby_key], bins=bins_array, include_lowest=True)

            if save_masks:
                outdir = Path(save_masks) / group_col
                outdir.mkdir(parents=True, exist_ok=True)

                # Vectorized creation of masks
                for interval in df[group_col].cat.categories:
                    mask = df[group_col].values == interval
                    fname = f"{group_col}_{interval.left}_{interval.right}.tif"
                    Raster.from_array(mask.reshape(shape), transform=transform, crs=crs).save(outdir / fname)

        elif isinstance(bins_array, Raster):
            data = np.asarray(bins_array.data.data).flatten()
            if len(data) != len(df):
                raise ValueError(f"Mask has invalid length {len(data)}, expected {len(df)}")
            # binary mask
            if bins_array.is_mask:
                df[group_col] = pd.Categorical(data, categories=[False, True])
            # segmentation
            else:
                df[group_col] = pd.Categorical(data)
        else:
            raise NotImplementedError("This type of bins does not yet work")

        # Compute the stats
        groupby_keys.append(group_col)

    # Use aggregation directly on categorical columns (fast)
    result = df.groupby(groupby_keys, observed=True, dropna=False)[list(aggregated_vars.keys())].agg(statistics)

    if save_csv:
        csv_file = Path(save_csv)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        result.reset_index().to_csv(csv_file, index=False)

    return result
