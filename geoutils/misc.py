"""Miscellaneous functions, mainly for testing."""
from __future__ import annotations

import functools
import warnings

import numpy as np
import rasterio as rio

import geoutils
from geoutils._typing import ArrayLike
from geoutils.georaster import Raster, RasterType


def array_equal(
    array1: RasterType | ArrayLike,
    array2: RasterType | ArrayLike,
    equal_nan: bool = True,
    tolerance: float = 0.0,
) -> bool:
    """
    Check if two arrays or Rasters are equal.

    This function mirrors (and partly uses) 'np.array_equal' with these exceptions:
        1. Different dtypes are okay as long as they are equal (e.g. '1 == 1.0' is True)
        2. Rasters are directly comparable.
        3. masked_array masks are respected.
        4. A tolerance argument is added.
        5. The function works with numpy<=1.18.

    :param array1: The first array-like object to compare.
    :param array2: The second array-like object to compare.
    :param equal_nan: Whether to compare NaNs as equal ('NaN == NaN' is True)
    :param tolerance: The maximum allowed summed difference between the arrays.

    Examples:
        Any object that can be parsed as an array can be compared.
        >>> arr1 = [1, 2, 3]
        >>> arr2 = np.array([1., 2., 3.])
        >>> array_equal(arr1, arr2)
        True

        Nans are equal by default, but can be disabled with 'equal_nan=False'
        >>> arr3 = np.array([1., 2., np.nan])
        >>> array_equal(arr1, arr3)
        False
        >>> array_equal(arr3, arr3.copy())
        True
        >>> array_equal(arr3, arr3, equal_nan=False)
        False

        The equality tolerance can be set with the 'tolerance' argument (defaults to 0).
        >>> arr4 = np.array([1., 2., 3.1])
        >>> array_equal(arr1, arr4)
        False
        >>> array_equal(arr1, arr4, tolerance=0.2)
        True

        Masks in masked_arrays are respected.
        >>> arr5 = np.ma.masked_array(arr1, [False, False, True])
        >>> array_equal(arr1, arr5)
        False
        >>> array_equal(arr3, arr5)
        True
        >>> array_equal(arr3, arr5, equal_nan=False)
        False
    """
    arrays: list[np.ndarray] = []
    strings_compared = False  # Flag to handle string arrays instead of numeric

    # Convert both inputs to numpy ndarrays
    for arr in array1, array2:
        if any(s in np.dtype(type(np.asanyarray(arr)[0])).name for s in ("<U", "str")):
            strings_compared = True
        if isinstance(arr, Raster):  # If a Raster subclass, take its data. I don't know why mypy complains here!
            arr = arr.data  # type: ignore
        if isinstance(arr, np.ma.masked_array):  # If a masked_array, replace the masked values with nans
            if "float" not in np.dtype(arr.dtype).name:
                arr = arr.astype(float)
            arrays.append(arr.filled(np.nan))  # type: ignore
        else:
            arrays.append(np.asarray(arr))

    if np.shape(arrays[0]) != np.shape(arrays[1]):
        return False

    if strings_compared:  # If they are strings, the tolerance/nan handling is irrelevant.
        return bool(np.array_equal(arrays[0], arrays[1]))

    diff = np.diff(arrays, axis=0)

    if "float" in np.dtype(diff.dtype).name and np.any(~np.isfinite(diff)):
        # Check that the nan-mask is equal. If it's not, or nans are not allowed at all, return False
        if not equal_nan or not np.array_equal(np.isfinite(arrays[0]), np.isfinite(arrays[1])):
            return False

    return bool(np.nansum(np.abs(diff)) <= tolerance)


def deprecate(removal_version: str | None = None, details: str | None = None):  # type: ignore
    """
    Trigger a DeprecationWarning for the decorated function.

    :param func: The function to be deprecated.
    :param removal_version: Optional. The version at which this will be removed.
                            If this version is reached, a ValueError is raised.
    :param details: Optional. A description for why the function was deprecated.

    :triggers DeprecationWarning: For any call to the function.

    :raises ValueError: If 'removal_version' was given and the current version is equal or higher.

    :returns: The decorator to decorate the function.
    """

    def deprecator_func(func):  # type: ignore
        @functools.wraps(func)
        def new_func(*args, **kwargs):  # type: ignore
            # True if it should warn, False if it should raise an error
            should_warn = removal_version is None or removal_version > geoutils.version.version

            # Add text depending on the given arguments and 'should_warn'.
            text = (
                f"Call to deprecated function '{func.__name__}'."
                if should_warn
                else f"Deprecated function '{func.__name__}' was removed in {removal_version}."
            )

            # Add the details explanation if it was given, and make sure the sentence is ended.
            if details is not None:
                details_frm = details.strip()
                if details_frm[0].islower():
                    details_frm = details_frm[0].upper() + details_frm[1:]

                text += " " + details_frm

                if not any(text.endswith(c) for c in ".!?"):
                    text += "."

            if should_warn and removal_version is not None:
                text += f" This functionality will be removed in version {removal_version}."
            elif not should_warn:
                text += f" Current version: {geoutils.version.version}."

            if should_warn:
                warnings.warn(text, category=DeprecationWarning, stacklevel=2)
            else:
                raise ValueError(text)

            return func(*args, **kwargs)

        return new_func

    return deprecator_func


def resampling_method_from_str(method_str: str) -> rio.warp.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.warp.Resampling:
        if str(method).replace("Resampling.", "") == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.warp.Resampling method. "
            f"Valid methods: {[str(method).replace('Resampling.', '') for method in rio.warp.Resampling]}"
        )
    return resampling_method
