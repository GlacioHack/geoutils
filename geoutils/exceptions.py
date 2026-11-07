# Copyright (c) 2026 GeoUtils developers
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

"""Errors and warning categories used by GeoUtils."""

from __future__ import annotations

import sys
import warnings
from types import FrameType


class GeoUtilsWarning(Warning):
    """Base category for warnings issued by GeoUtils."""


class GeoUtilsUserWarning(GeoUtilsWarning, UserWarning):
    """Warning about a GeoUtils operation or its result."""


class GeoUtilsDeprecationWarning(GeoUtilsWarning, DeprecationWarning):
    """Warning about a deprecated GeoUtils API."""


def _warn(message: str, category: type[GeoUtilsWarning] = GeoUtilsUserWarning) -> None:
    """Emit a warning at the first caller outside GeoUtils, skipping internal helpers and decorators."""

    # Skip stack frames so the warning points to the code that called GeoUtils
    # Count them dynamically because internal helpers and decorators can change the call depth
    frame: FrameType | None = sys._getframe(1)
    stacklevel = 2
    try:
        while frame is not None and frame.f_globals.get("__name__", "").split(".")[0] == "geoutils":
            frame = frame.f_back
            stacklevel += 1
    finally:
        del frame

    warnings.warn(message, category=category, stacklevel=stacklevel)


class GeoUtilsInputError(ValueError):
    """Base error for invalid inputs to GeoUtils operations."""


class GeoUtilsMismatchError(GeoUtilsInputError):
    """Base error for inputs that do not match each other."""


class CRSMismatchError(GeoUtilsMismatchError):
    """Raised when input coordinate reference systems differ."""


class GridMismatchError(GeoUtilsMismatchError):
    """Raised when input raster grids differ."""


class PointsMismatchError(GeoUtilsMismatchError):
    """Raised when input point coordinates or their order differ."""


class ShapeMismatchError(GeoUtilsMismatchError):
    """Raised when input array shapes differ."""


class InvalidBoundsError(GeoUtilsInputError):
    """Raised when bound-type input is not recognized."""


class InvalidPointsError(GeoUtilsInputError):
    """Raised when point-type input is not recognized."""


class InvalidCRSError(GeoUtilsInputError):
    """Raised when CRS-type input is not recognized."""


class InvalidGridError(GeoUtilsInputError):
    """Raised when grid-type input is not recognized."""


class InvalidResolutionError(GeoUtilsInputError):
    """Raised when resolution-type input is not recognized."""


class InvalidShapeError(GeoUtilsInputError):
    """Raised when shape input is invalid."""


class InvalidNodataError(GeoUtilsInputError):
    """Raised when nodata values or handling rules are invalid."""


class InvalidResamplingError(GeoUtilsInputError):
    """Raised when a resampling method or its configuration is invalid."""


class NoValidDataError(GeoUtilsInputError):
    """Raised when an operation has too few valid observations."""


class IgnoredGridWarning(GeoUtilsUserWarning):
    """Raised when grid-type input is ignored (because redundant with others)."""
