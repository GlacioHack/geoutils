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
from scipy.stats.mstats import mquantiles

from geoutils import profiler
from geoutils._typing import NDArrayNum
from geoutils.stats.estimators import linear_error, nmad, rmse, sum_square

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
    "standard_deviation": "Standard deviation",
    "validcount": "Valid count",
    "valid_count": "Valid count",
    "totalcount": "Total count",
    "total_count": "Total count",
    "percentagevalidpoints": "Percentage valid points",
    "percentage_valid_points": "Percentage valid points",
}  # type: ignore

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

_ALIAS_STATS_LIST_MASK = {
    "validinliercount": "Valid inlier count",
    "valid_inlier_count": "Valid inlier count",
    "totalinliercount": "Total inlier count",
    "total_inlier_count": "Total inlier count",
    "percentagevalidinlierpoints": "Percentage valid inlier points",
    "percentage_valid_inlier_points": "Percentage valid inlier points",
    "percentageinlierpoints": "Percentage inlier points",
    "percentage_inlier_points": "Percentage inlier points",
}


@profiler.profile("geoutils.stats.stats._statistics", memprof=True)  # type: ignore
def _statistics(
    data: NDArrayNum,
    stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | None = None,
    counts: tuple[int, int] | None = None,
) -> dict[str, float]:
    """
    Calculate common statistics for an N-D array :

    - Mean: arithmetic mean of the data, ignoring masked values.
    - Median: middle value when the valid data points are sorted in increasing order, ignoring masked values.
    - Max: maximum value among the data, ignoring masked values.
    - Min: minimum value among the data, ignoring masked values.
    - Sum: sum of all data, ignoring masked values.
    - Sum of squares: sum of the squares of all data, ignoring masked values.
    - 90th percentile: point below which 90% of the data falls, ignoring masked values.
    - IQR (Interquartile Range): difference between the 75th and 25th percentile of a dataset, ignoring masked values.
    - LE90 (Linear Error with 90% confidence): difference between the 95th and 5th percentiles of a dataset, \
    representing the range within which 90% of the data points lie. Ignore masked values.
    - NMAD (Normalized Median Absolute Deviation): robust measure of variability in the data, less sensitive to \
    outliers compared to standard deviation. Ignore masked values.
    - RMSE (Root Mean Square Error): commonly used to express the magnitude of errors or variability and can give \
    insight into the spread of the data. Only relevant when the raster represents a difference of two objects. \
    Ignore masked values.
    - Std (Standard deviation): measures the spread or dispersion of the data around the mean, ignoring masked values.
    - Valid count: number of finite data points in the array. It counts the non-masked elements.
    - Total count: total size of the raster.
    - Percentage valid points: ratio between Valid count and Total count.

    For all statistics up to and including "Std", NumPy Masked functions are used (directly or in the calculation)
    in case of a masked array, NumPy module otherwise.

    "Valid count" represents all non zero and not masked pixels in the input data (final_count_nonzero), previously
    calculated in case of a Raster.get_stats() called with an inlier_mask, before the mask application. NumPy Masked
    functions is used is this case or if the input was already a masked array.
    Percentage valid points is calculated accordingly.

    If an inlier mask is passed:
    - Total inlier count: number of data points in the inlier mask.
    - Valid inlier count: number of unmasked data points in the array after applying the inlier mask.
    - Percentage inlier points: ratio between Valid inlier count and Valid count. Useful for classification statistics.
    - Percentage valid inlier points: ratio between Valid inlier count and Total inlier count.

    They are all computed based on the previously stated final_count_nonzero.

    Callable functions are supported as well.

    :param data: Array on which to compute statistics.
    :param stats_name: list of names of the statistics to retrieve. If None, all statistics are returned.
            Accepted names include:
            `mean`, `median`, `max`, `min`, `sum`, `sum of squares`, `90th percentile`, `iqr`, `LE90`, `nmad`, `rmse`,
            `std`, `valid count`, `total count`, `percentage valid points` and if an inlier mask is passed :
            `valid inlier count`, `total inlier count`, `percentage inlier point`, `percentage valid inlier points`.
            Custom callables can also be provided.
    :param counts: Tuple with number of finite data points in array and number of valid points in inlier_mask.

    :returns: A dictionary containing the calculated statistics for the selected band.
    """

    if np.ma.isMaskedArray(data):

        # Count non zero and not masked pixels in the input data
        final_count_nonzero = np.count_nonzero(~np.ma.getmaskarray(data))

        # Compute valid count from non zero and not masked pixels in the input data
        # beforehand saved in counts[0] in case of a inler_mask parameter in get_stats()
        valid_count = final_count_nonzero if counts is None else counts[0]

        stats_dict = {
            "Mean": np.ma.mean,
            "Median": np.ma.median,
            "Max": np.ma.max,
            "Min": np.ma.min,
            "Sum": np.ma.sum,
            "Sum of squares": sum_square,
            "90th percentile": partial(lambda x: mquantiles(x, prob=0.9, alphap=1, betap=1)[0]),
            "LE90": partial(linear_error, interval=90),
            "IQR": partial(iqr, nan_policy="omit"),  # ignore masked value (nan),
            "NMAD": nmad,
            "RMSE": rmse,
            "Standard deviation": np.ma.std,
        }  # type: ignore

    else:
        # Count non zero pixels in the input data
        final_count_nonzero = np.count_nonzero(np.isfinite(data))

        # Compute valid count from non zero and not masked pixels in the input data
        # beforehand saved in counts[0] in case of a inler_mask parameter in get_stats()
        valid_count = final_count_nonzero if counts is None else counts[0]

        stats_dict = {
            "Mean": np.nanmean,
            "Median": np.nanmedian,
            "Max": np.nanmax,
            "Min": np.nanmin,
            "Sum": np.nansum,
            "Sum of squares": sum_square,
            "90th percentile": partial(np.nanpercentile, q=90),
            "LE90": partial(linear_error, interval=90),
            "IQR": partial(iqr, nan_policy="omit"),  # ignore masked value (nan),
            "NMAD": nmad,
            "RMSE": rmse,
            "Standard deviation": np.nanstd,
        }  # type: ignore

    # Pixels counts
    stats_dict.update(
        {
            "Valid count": valid_count,
            "Total count": data.size,
            "Percentage valid points": (valid_count / data.size) * 100 if data.size else np.nan,
        }
    )

    # If there are no valid data points, set all statistics to NaN
    if final_count_nonzero == 0:
        logging.warning("Empty raster, returns Nan for all stats")
        if stats_name is None:
            stat_data_valid = STATS_LIST  # type: ignore
        else:
            stat_data_valid = stats_name  # type: ignore
        res_dict = {
            stat_name: (
                stats_dict[stat_name] if (stat_name in STATS_LIST and not callable(stats_dict[stat_name])) else np.nan
            )
            for stat_name in stat_data_valid
        }  # type: ignore

        if stats_name is not None:
            stat_with_alias = set(stats_name).intersection(list(_STATS_ALIASES.keys()))  # type: ignore
            if stat_with_alias:
                for stat_name in stat_with_alias:
                    alias = _STATS_ALIASES[stat_name]  # type: ignore
                    if callable(stats_dict[alias]):
                        res_dict[stat_name] = np.nan  # type: ignore
                    else:
                        res_dict[stat_name] = stats_dict[alias]  # type: ignore

    else:
        if stats_name is None:
            res_dict = stats_dict  # type: ignore
            if stats_name is None:
                for key in stats_dict.keys():
                    if callable(stats_dict[key]):
                        res_dict[key] = stats_dict[key](data)  # type: ignore
        else:
            res_dict = {}  # type: ignore
            for stat_name in stats_name:

                # Compute stat if in stats_dict keys
                if isinstance(stat_name, str) and stat_name in stats_dict.keys():
                    if callable(stats_dict[stat_name]):
                        res_dict[stat_name] = stats_dict[stat_name](data)  # type: ignore
                    else:
                        res_dict[stat_name] = stats_dict[stat_name]  # type: ignore
                # Compute stat if in _STATS_ALIASES keys
                elif isinstance(stat_name, str) and stat_name in _STATS_ALIASES.keys():
                    if callable(stats_dict[_STATS_ALIASES[stat_name]]):
                        res_dict[stat_name] = stats_dict[_STATS_ALIASES[stat_name]](data)  # type: ignore
                    else:
                        res_dict[stat_name] = stats_dict[_STATS_ALIASES[stat_name]]  # type: ignore
                # Compute stat if callable
                elif callable(stat_name):
                    res_dict[stat_name.__name__] = stat_name(data)  # type: ignore

                # Stat not recognized
                else:
                    logging.warning("Statistic name '%s' is not recognized", stat_name)
                    res_dict[stat_name] = np.float32(np.nan)  # type: ignore

    # If inlier mask parameter given before in get_stats() and if one of these stats is wanted
    if counts is not None and (
        stats_name is None
        or list(
            set(STATS_LIST_MASK).intersection(stats_name) or list(set(_ALIAS_STATS_LIST_MASK).intersection(stats_name))
        )
    ):
        dict_c = {
            "Valid inlier count": final_count_nonzero,
            "Total inlier count": counts[1],
            "Percentage inlier points": (final_count_nonzero / counts[0]) * 100,
            "Percentage valid inlier points": (final_count_nonzero / counts[1]) * 100 if counts[1] != 0 else 0,
        }

        if stats_name is None:
            # Add all stats
            res_dict.update(dict_c)
        else:
            # Add all stats if the name is on the list
            res_dict.update({k: dict_c[k] for k in list(set(STATS_LIST_MASK).intersection(stats_name))})
            # and the stats from alias stats
            res_dict.update(
                {
                    k: dict_c[_ALIAS_STATS_LIST_MASK[k]]
                    for k in list(set(_ALIAS_STATS_LIST_MASK).intersection(stats_name))
                }
            )
    return res_dict  # type: ignore
