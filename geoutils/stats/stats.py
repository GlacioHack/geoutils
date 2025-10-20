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
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from scipy.stats import iqr

from geoutils._typing import NDArrayNum
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

STATS_LIST = [
    "Mean",
    "Median",
    "Max",
    "Min",
    "Sum",
    "Sum of squares",
    "90th percentile",
    "IQR",
    "LE90",
    "NMAD",
    "RMSE",
    "Standard deviation",
    "Valid count",
    "Total count",
    "Percentage valid points",
]

STATS_LIST_MASK = [
    "Valid inlier count",
    "Total inlier count",
    "Percentage inlier points",
    "Percentage valid inlier points",
]


def _my_statistics_partial(
    data: NDArrayNum,
    stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | None = None,
    counts: tuple[int, int] | None = None,
    mask_count_nonzero: int = None,
) -> dict[str, float]:
    if np.ma.isMaskedArray(data):
        mask = ~np.ma.getmaskarray(data)
        mdata = np.ma.filled(data.astype(float), np.nan)
    else:
        mask = np.isfinite(data)
        mdata = data

    if isinstance(stats_name, str):
        stats_name = [stats_name]

    # If there are no valid data points, set all statistics to NaN
    if mask_count_nonzero == 0:
        print(stats_name)
        logging.warning("Empty raster, returns Nan for all stats")
        if stats_name is None:
            res_dict = {stat_name: np.nan for stat_name in STATS_LIST + STATS_LIST_MASK}  # type: ignore
        else:
            res_dict = {stat_name: np.nan for stat_name in stats_name}  # type: ignore

    else:
        # Valid count
        valid_count = np.count_nonzero(mask) if counts is None else counts[0]

        stats_dict = {
            "Mean": partial(np.ma.mean, data),
            "Median": partial(np.ma.median, data),
            "Max": partial(np.ma.max, data),
            "Min": partial(np.ma.min, data),
            "Sum": partial(np.ma.sum, data),
            "Sum of squares": partial(lambda x: np.ma.sum(np.square(x)), data),
            "90th percentile": partial(lambda x: np.nanpercentile(x, 90), mdata),
            "LE90": partial(linear_error, mdata, interval=90),
            "IQR": partial(iqr, mdata, nan_policy="omit"),  # ignore masked value (nan),
            "NMAD": partial(nmad, data),
            "RMSE": partial(lambda x: np.sqrt(np.ma.mean(np.square(x))), data),
            "Standard deviation": partial(np.ma.std, data),
            "Valid count": partial(lambda x: x, valid_count),
            "Total count": partial(lambda x: x.size, data),
            "Percentage valid points": partial(lambda x: (valid_count / x.size) * 100, data),
        }

        res_dict = {}
        if stats_name is None:
            res_dict = {k: stats_dict[k]() for k in stats_dict.keys() if callable(stats_dict[k])}
        else:
            for stat_name in stats_name:
                if isinstance(stat_name, str) and stat_name in stats_dict.keys() and callable(stats_dict[stat_name]):
                    res_dict[stat_name] = stats_dict[stat_name]()  # type: ignore
                elif (
                    isinstance(stat_name, str)
                    and stat_name in _STATS_ALIASES.keys()
                    and callable(stats_dict[_STATS_ALIASES[stat_name]])
                ):
                    res_dict[stat_name] = stats_dict[_STATS_ALIASES[stat_name]]()  # type: ignore
                elif callable(stat_name):
                    res_dict[stat_name.__name__] = stat_name(data)  # type: ignore
                else:
                    logging.warning("Statistic name '%s' is not recognized", stat_name)
                    res_dict[stat_name] = np.float32(np.nan)  # type: ignore

        list_counts_stats = [
            "Valid inlier count",
            "Total inlier count",
            "Percentage inlier points",
            "Percentage valid inlier points",
        ]

        # If inlier mask was passed
        if counts is not None and (stats_name is None or list(set(list_counts_stats).intersection(stats_name))):
            valid_inlier_count = np.count_nonzero(mask)
            dict_c = {
                "Valid inlier count": valid_inlier_count,
                "Total inlier count": counts[1],
                "Percentage inlier points": (valid_inlier_count / counts[0]) * 100,
                "Percentage valid inlier points": (valid_inlier_count / counts[1]) * 100 if counts[1] != 0 else 0,
            }

            if stats_name is None:
                res_dict.update(dict_c)

    return res_dict


"""def _statistics(data: NDArrayNum, counts: tuple[int, int] | None = None) -> dict[str, np.floating[Any]]:
    ""
    Calculate common statistics for an N-D array.

    :param data: Array on which to compute statistics.
    :param counts: Tuple with number of finite data points in array and number of valid points in inlier_mask.

    :returns: A dictionary containing the calculated statistics for the selected band.
    ""

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

    return stats_dict"""
