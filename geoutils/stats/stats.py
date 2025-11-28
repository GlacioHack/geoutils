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

"""Module for zonal statistics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import iqr

from geoutils._typing import NDArrayNum
import geoutils.profiler as Profiler
from geoutils.stats.estimators import linear_error, nmad

_STATS_ALIASES = {
    "mean": "Mean",
    "median": "Median",
    "max": "Max",
    "maximum": "Max",
    "min": "Min",
    "minimum": "Min",
    "sum": "Sum",
    "sumofsquares": "Sum of squares",
    "sum2": "Sum of squares",
    "90thpercentile": "90th percentile",
    "90percentile": "90th percentile",
    "iqr": "IQR",
    "le90": "LE90",
    "nmad": "NMAD",
    "rmse": "RMSE",
    "rms": "RMSE",
    "std": "Standard deviation",
    "standarddeviation": "Standard deviation",
    "validcount": "Valid count",
    "totalcount": "Total count",
    "percentagevalidpoints": "Percentage valid points",
    "validinliercount": "Valid inlier count",
    "totalinliercount": "Total inlier count",
    "percentagevalidinlierpoints": "Percentage valid inlier points",
    "percentageinlierpoints": "Percentage inlier points",
}


@Profiler.profile("stats.stats._statistics", memprof=True)  # type: ignore
def _statistics(data: NDArrayNum, counts: tuple[int, int] | None = None) -> dict[str, np.floating[Any]]:
    """
    Calculate common statistics for an N-D array.

    :param data: Array on which to compute statistics.
    :param counts: Tuple with number of finite data points in array and number of valid points in inlier_mask.

    :returns: A dictionary containing the calculated statistics for the selected band.
    """

    # Pre-computing depending on nature of array
    # TODO: Array is duplicated into filled array with NaN at every call, doubling memory usage
    if np.ma.isMaskedArray(data):
        mask = ~np.ma.getmaskarray(data)
        mdata = np.ma.filled(data.astype(float), np.nan)
    else:
        mask = np.isfinite(data)
        mdata = data
    # Valid count
    valid_count = np.count_nonzero(mask) if counts is None else counts[0]
    # Other stats
    stats_dict = {
        "Mean": np.ma.mean(data),
        "Median": np.ma.median(data),
        "Max": np.ma.max(data),
        "Min": np.ma.min(data),
        "Sum": np.ma.sum(data),
        "Sum of squares": np.ma.sum(np.square(data)),
        "90th percentile": np.nanpercentile(mdata, 90),
        "LE90": linear_error(mdata, interval=90),
        "IQR": iqr(mdata, nan_policy="omit"),  # ignore masked value (nan),
        "NMAD": nmad(data),
        "RMSE": np.sqrt(np.ma.mean(np.square(data))),
        "Standard deviation": np.ma.std(data),
        "Valid count": valid_count,
        "Total count": data.size,
        "Percentage valid points": (valid_count / data.size) * 100,
    }

    # If inlier mask was passed
    if counts is not None:
        valid_inlier_count = np.count_nonzero(mask)
        stats_dict.update(
            {
                "Valid inlier count": valid_inlier_count,
                "Total inlier count": counts[1],
                "Percentage inlier points": (valid_inlier_count / counts[0]) * 100,
                "Percentage valid inlier points": (valid_inlier_count / counts[1]) * 100 if counts[1] != 0 else 0,
            }
        )

    # If there are no valid data points, set all statistics to NaN
    if np.count_nonzero(mask) == 0:
        logging.warning("Empty raster, returns Nan for all stats")
        for key in stats_dict:
            stats_dict[key] = np.nan

    return stats_dict


def _get_single_stat(
    stats_dict: dict[str, np.floating[Any]], stats_aliases: dict[str, str], stat_name: str
) -> np.floating[Any]:
    """
    Retrieve a single statistic based on a flexible name or alias.

    :param stats_dict: The dictionary of available statistics.
    :param stats_aliases: The dictionary of alias mappings to the actual stat names.
    :param stat_name: The name or alias of the statistic to retrieve.

    :returns: The requested statistic value, or None if the stat name is not recognized.
    """

    normalized_name = stat_name.lower().replace(" ", "").replace("_", "").replace("-", "")
    if normalized_name in stats_aliases:
        actual_name = stats_aliases[normalized_name]
        return stats_dict[actual_name]
    else:
        logging.warning("Statistic name '%s' is not recognized", stat_name)
        return np.float32(np.nan)
