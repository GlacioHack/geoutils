"""Typing aliases for internal use."""
from __future__ import annotations

import sys
from typing import Any, List, Tuple, Union

import numpy as np

# Mypy has issues with the builtin Number type (https://github.com/python/mypy/issues/3186)
Number = int | float | np.integer[Any] | np.floating[Any]

# Only for Python >= 3.9
if sys.version_info.minor >= 9:

    from numpy.typing import (  # this syntax works starting on Python 3.9
        ArrayLike,
        DTypeLike,
        NDArray,
    )

    # Simply define here if they exist (for backwards compatibility below)
    DTypeLike = DTypeLike
    ArrayLike = ArrayLike

    # Use NDArray wrapper to easily define numerical (float or int) N-D array types, and boolean N-D array types
    NDArrayNum = NDArray[np.floating[Any] | np.integer[Any]]
    NDArrayBool = NDArray[np.bool[Any]]
    # Define numerical (float or int) masked N-D array type
    MArrayNum = np.ma.masked_array[Any, np.dtype[np.floating[Any] | np.integer[Any]]]
    MArrayBool = np.ma.masked_array[Any, np.dtype[np.bool[Any]]]

else:

    # Make an array-like type (since the array-like numpy type only exists in numpy>=1.20)
    ArrayLike = Union[np.ndarray, np.ma.masked_array, List[Any], Tuple[Any]]  # type: ignore
    DTypeLike = Union[str, type, np.dtype]  # type: ignore

    NDArrayNum = np.ndarray  # type: ignore
    NDArrayBool = np.ndarray  # type: ignore
    MArrayNum = np.ma.masked_array  # type: ignore
