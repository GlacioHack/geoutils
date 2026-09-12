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

"""Module for additional statistical estimators."""

from typing import Any

import numpy as np
from scipy.stats.mstats import mquantiles

from geoutils._typing import NDArrayNum


def nmad(data: NDArrayNum, nfact: float = 1.4826) -> np.floating[Any]:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.
    Default scaling factor is 1.4826 to scale the median absolute deviation (MAD) to the dispersion of a normal
    distribution (see https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation, and
    e.g. Höhle and Höhle (2009), http://dx.doi.org/10.1016/j.isprsjprs.2009.02.003)

    :param data: Input array or raster.
    :param nfact: Normalization factor for the data.

    :returns nmad: (Normalized) median absolute deviation of data.
    """
    if isinstance(data, np.ma.masked_array):
        return nfact * np.ma.median(np.abs(data - np.ma.median(data)))
    else:
        return nfact * np.nanmedian(np.abs(data - np.nanmedian(data)))


def linear_error(data: NDArrayNum, interval: float = 90) -> np.floating[Any]:
    """
    Compute the linear error (LE) for a given dataset, representing the difference between the upper and lower
    percentiles of the data. By default, this calculates the central interval containing 90% of the values (LE90).

    :param data: A numpy array or masked array of data, typically representing the differences (errors) in elevation or
        another quantity.
    :param interval: The central interval to compute, specified as a percentage. For example, an interval of 90 will
        compute the range between the 5th and 95th percentiles (LE90). This value must be greater than 0 and at most 100.

    :returns: The computed linear error, which is the difference between the upper and lower percentiles.

    :raises ValueError: If interval is not greater than 0 and at most 100.
    """
    # Validate the interval
    if not (0 < interval <= 100):
        raise ValueError("Interval must be between 0 and 100")

    max = 50 + interval / 2
    min = 50 - interval / 2
    if isinstance(data, np.ma.masked_array):
        return (
            mquantiles(data, prob=max / 100, alphap=1, betap=1) - mquantiles(data, prob=min / 100, alphap=1, betap=1)
        )[0]
    else:
        return np.nanpercentile(data, max) - np.nanpercentile(data, min)


def sum_square(data: NDArrayNum) -> np.floating[Any]:
    """
    Calculate the sum of the square of a data array.

    :param data: A numpy array or masked array of data, typically representing the differences (errors) in elevation or
        another quantity.
    :return: sum square
    """
    # Promote integer values before squaring so their original storage type cannot overflow
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64)
    if np.ma.isMaskedArray(data):
        return np.ma.sum(np.square(data))
    else:
        return np.nansum(np.square(data))


def rmse(data: NDArrayNum) -> np.floating[Any]:
    """
    Calculate the RMSE of a data array.

    :param data: A numpy array or masked array of data, typically representing the differences (errors) in elevation or
        another quantity.
    :return: rmse
    """
    # Promote integer values before squaring so their original storage type cannot overflow
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64)
    if np.ma.isMaskedArray(data):
        return np.sqrt(np.ma.mean(np.square(data)))
    else:
        return np.sqrt(np.nanmean(np.square(data)))
