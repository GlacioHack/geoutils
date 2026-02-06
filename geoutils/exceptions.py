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

"""Functions for consistent input checks and dispatching."""

from __future__ import annotations


class InvalidBoundsError(ValueError):
    """Raised when bound-type input is not recognized."""


class InvalidPointsError(ValueError):
    """Raised when point-type input is not recognized."""


class InvalidCRSError(ValueError):
    """Raised when CRS-type input is not recognized."""


class InvalidGridError(ValueError):
    """Raised when grid-type input is not recognized."""


class InvalidResolutionError(ValueError):
    """Raised when resolution-type input is not recognized."""


class InvalidShapeError(ValueError):
    """Raised when resolution-type input is not recognized."""


class IgnoredGridWarning(UserWarning):
    """Raised when grid-type input is ignored (because redundant with others)."""
