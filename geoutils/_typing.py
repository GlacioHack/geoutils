"""Typing aliases for internal use."""
from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np

# Make an array-like type (since the array-like numpy type only exists in numpy>=1.20)
ArrayLike = Union[np.ndarray, np.ma.masked_array, List[Any], Tuple[Any]]

DTypeLike = Union[str, type, np.dtype]

# Mypy has issues with the builtin Number type (https://github.com/python/mypy/issues/3186)
AnyNumber = Union[int, float, np.number]
