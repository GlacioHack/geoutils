""" Statistical tools"""

from typing import Any

import numpy as np

from geoutils._typing import NDArrayNum


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
        data_arr = data.compressed()
    else:
        data_arr = np.asarray(data)
    return nfact * np.nanmedian(np.abs(data_arr - np.nanmedian(data_arr)))
