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

""" Statistical tools"""
from typing import Any

import numpy as np

from geoutils._typing import NDArrayNum
from geoutils.raster.array import get_array_and_mask


def nmad(data: NDArrayNum, nfact: float = 1.4826) -> np.floating[Any]:
    """
    Calculate the normalized median absolute deviation (NMAD) of an array.
    Default scaling factor is 1.4826 to scale the median absolute deviation (MAD) to the dispersion of a normal
    distribution (see https://en.wikipedia.org/wiki/Median_absolute_deviation#Relation_to_standard_deviation, and
    e.g. Höhle and Höhle (2009), http://dx.doi.org/10.1016/j.isprsjprs.2009.02.003)

    :param data: Input array or raster
    :param nfact: Normalization factor for the data

    :returns nmad: (normalized) median absolute deviation of data.
    """
    if isinstance(data, np.ma.masked_array):
        data_arr = get_array_and_mask(data)[0]
    else:
        data_arr = np.asarray(data)
    return nfact * np.nanmedian(np.abs(data_arr - np.nanmedian(data_arr)))


def linear_error(data: NDArrayNum, interval: float = 90) -> np.floating[Any]:
    """
    Compute the linear error (LE) for a given dataset, representing the range of differences between the upper and
    lower percentiles of the data. By default, this calculates the 90% confidence interval (LE90).

    :param data: A numpy array or masked array of data, typically representing the differences (errors) in elevation or
    another quantity.
    :param interval: The confidence interval to compute, specified as a percentage. For example, an interval of 90 will
    compute the range between the 5th and 95th percentiles (LE90). This value must be between 0 and 100.

    return: The computed linear error, which is the difference between the upper and lower percentiles.

    raises: ValueError if the `interval` is not between 0 and 100.
    """
    # Validate the interval
    if not (0 < interval <= 100):
        raise ValueError("Interval must be between 0 and 100")

    if isinstance(data, np.ma.masked_array):
        mdata = np.ma.filled(data.astype(float), np.nan)
    else:
        mdata = data
    le = np.nanpercentile(mdata, 50 + interval / 2) - np.nanpercentile(mdata, 50 - interval / 2)
    return le
