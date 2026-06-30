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

import warnings
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from scipy.stats import iqr
from scipy.stats.mstats import mquantiles

from geoutils import profiler
from geoutils._typing import NDArrayNum
from geoutils.stats.estimators import linear_error, nmad, rmse, sum_square

_STATS_ALIAS_OP = {
    "mean": "Mean",
    "median": "Median",
    "max": "Max",
    "min": "Min",
    "sum": "Sum",
    "sumofsquares": "Sum of squares",
    "90thpercentile": "90th percentile",
    "iqr": "IQR",
    "le90": "LE90",
    "nmad": "NMAD",
    "rmse": "RMSE",
    "std": "Standard deviation",
}

_STATS_ALIAS_COUNTS = {
    "validcount": "Valid count",
    "totalcount": "Total count",
    "percentagevalidpoints": "Percentage valid points",
}

_STATS_ALIAS_GEN = _STATS_ALIAS_OP | _STATS_ALIAS_COUNTS

_SYNONYMES = {
    "maximum": "max",
    "minimum": "min",
    "sum": "Sum",
    "sum2": "sumofsquares",
    "90percentile": "90thpercentile",
    "rms": "rmse",
    "standarddeviation": "std",
}

_STATS_ALIAS_MASK = {
    "validinliercount": "Valid inlier count",
    "totalinliercount": "Total inlier count",
    "percentagevalidinlierpoints": "Percentage valid inlier points",
    "percentageinlierpoints": "Percentage inlier points",
}  # type: ignore


_STATS_ALIAS_ALL = _STATS_ALIAS_GEN | _STATS_ALIAS_MASK
_ALIAS_STATS_GEN = {v: k for k, v in _STATS_ALIAS_GEN.items()}
_ALIAS_STATS_MASK = {v: k for k, v in _STATS_ALIAS_MASK.items()}
_ALIAS_STATS_ALL = _ALIAS_STATS_GEN | _ALIAS_STATS_MASK


_STATS_LIST_MIN = [
    "min",
    "max",
    "mean",
    "median",
    "std",
    "nmad",
    "validcount",
    "totalcount",
    "percentagevalidpoints",
]


@profiler.profile("geoutils.stats.stats._statistics", memprof=True)
def _statistics(
    data: NDArrayNum,
    stats_name: list[str | Callable[[NDArrayNum], np.floating[Any]]] | str | None = None,
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
            `valid inlier count`, `total inlier count`, `percentage inlier points`, `percentage valid inlier points`.
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
            "mean": np.ma.mean,
            "median": np.ma.median,
            "max": np.ma.max,
            "min": np.ma.min,
            "sum": np.ma.sum,
            "sumofsquares": sum_square,
            "90thpercentile": partial(lambda x: mquantiles(x, prob=0.9, alphap=1, betap=1)[0]),
            "le90": partial(linear_error, interval=90),
            "iqr": partial(iqr, nan_policy="omit"),  # ignore masked value (nan),
            "nmad": nmad,
            "rmse": rmse,
            "std": np.ma.std,
        }  # type: ignore

    else:
        # Count non zero pixels in the input data
        final_count_nonzero = np.count_nonzero(np.isfinite(data))

        # Compute valid count from non zero and not masked pixels in the input data
        # beforehand saved in counts[0] in case of a inler_mask parameter in get_stats()
        valid_count = final_count_nonzero if counts is None else counts[0]

        stats_dict = {
            "mean": np.nanmean,
            "median": np.nanmedian,
            "max": np.nanmax,
            "min": np.nanmin,
            "sum": np.nansum,
            "sumofsquares": sum_square,
            "90thpercentile": partial(np.nanpercentile, q=90),
            "le90": partial(linear_error, interval=90),
            "iqr": partial(iqr, nan_policy="omit"),  # ignore masked value (nan),
            "nmad": nmad,
            "rmse": rmse,
            "std": np.nanstd,
        }  # type: ignore

    # Pixels counts
    stats_dict.update(
        {
            "validcount": valid_count,
            "totalcount": data.size,
            "percentagevalidpoints": (valid_count / data.size) * 100 if data.size else np.nan,
        }
    )

    if counts is not None:
        stats_dict.update(
            {
                "validinliercount": final_count_nonzero,
                "totalinliercount": counts[1],
                "percentageinlierpoints": (final_count_nonzero / counts[0]) * 100,
                "percentagevalidinlierpoints": (final_count_nonzero / counts[1]) * 100 if counts[1] != 0 else 0,
            }
        )

    def get_stat_common_name(stat_name: str) -> str | None:
        if stat_name in stats_dict.keys():
            return stat_name
        else:
            for split_v in [None, "_"]:
                if "".join(stat_name.lower().split(split_v)) in _STATS_ALIAS_ALL.keys():
                    return "".join(stat_name.lower().split(split_v))
                if "".join(stat_name.lower().split(split_v)) in _ALIAS_STATS_ALL.keys():
                    return _ALIAS_STATS_GEN["".join(stat_name.lower().split(split_v))]
                elif "".join(stat_name.lower().split(split_v)) in _SYNONYMES:
                    return _SYNONYMES["".join(stat_name.lower().split(split_v))]

            return None

    def create_list(counts_is_none: bool, stats_name: str | None) -> list[str]:
        if stats_name is None:
            stat_names_res = _STATS_LIST_MIN
        else:
            stat_names_res = list(_STATS_ALIAS_GEN)
            if counts_is_none:
                stat_names_res = stat_names_res + list(_STATS_ALIAS_MASK.keys())
        return stat_names_res

    # If there are no valid data points, set all statistics to NaN
    if final_count_nonzero == 0:
        warnings.warn("Empty raster, returns Nan for all stats", category=UserWarning)
        alias = False
        if stats_name is None or stats_name == "all":
            stat_names_res = create_list(counts is not None, stats_name)  # type: ignore
            alias = True
        res_dict = {
            (_STATS_ALIAS_ALL[stat_name] if alias else stat_name): (
                stats_dict[get_stat_common_name(stat_name)]  # type: ignore
                if (
                    get_stat_common_name(stat_name) is not None
                    and not callable(stats_dict[get_stat_common_name(stat_name)])  # type: ignore
                )
                else np.nan
            )
            for stat_name in stat_names_res
        }  # type: ignore

    else:
        if stats_name is None or stats_name == "all":
            stat_names_res = create_list(counts is not None, stats_name)  # type: ignore

            res_dict = {
                _STATS_ALIAS_ALL[stat_name]: (
                    stats_dict[stat_name](data)  # type: ignore
                    if (callable(stats_dict[stat_name]))
                    else stats_dict[stat_name]
                )
                for stat_name in stat_names_res
            }  # type: ignore

        else:
            res_dict = {}  # type: ignore
            for stat_name in stats_name:

                # Compute stat if in stats_dict keys
                if isinstance(stat_name, str):
                    stat_common_name = get_stat_common_name(stat_name)
                    if stat_common_name:
                        if stat_common_name in stats_dict:
                            res_dict[stat_name] = stats_dict[stat_common_name]
                            if callable(res_dict[stat_name]):
                                res_dict[stat_name] = res_dict[stat_name](data)  # type: ignore
                        else:
                            res_dict[stat_name] = np.nan
                    else:
                        warnings.warn("Statistic name " + stat_name + " is not recognized", category=UserWarning)
                        res_dict[stat_name] = np.float32(np.nan)  # type: ignore

                # Compute stat if callable
                elif callable(stat_name):
                    res_dict[stat_name.__name__] = stat_name(data)  # type: ignore

                else:
                    warnings.warn("Statistic name " + stat_name + " is not recognized", category=UserWarning)
                    res_dict[stat_name] = np.float32(np.nan)  # type: ignore

    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in res_dict.items()}  # type: ignore
