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

"""Functions for consistent input checks and dispatching."""

from __future__ import annotations

from typing import Any, Sequence

def get_geo_attr(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """Retrieve an attribute from an object, or one of its accessors."""

    # Try direct attribute (Raster, Vector, PointCloud, or accessor class)
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)

    # Try accessors (rst, vct, pc)
    for accessor_name in accessors:
        accessor = getattr(obj, accessor_name, None)
        if accessor is not None and hasattr(accessor, attr_name):
            return getattr(accessor, attr_name)

    # Fallback
    raise AttributeError(
        f"Attribute '{attr_name}' not found on object {type(obj)} "
        f"or its potential accessors {accessors}."
    )

def has_geo_attr(obj: Any, attr_name: str, accessors: Sequence[str] = ("rst", "vct", "pc")) -> Any:
    """Check if attribute exists for an object, or one of its accessors."""

    # Check direct attribute (Raster, Vector, PointCloud, or accessor class)
    if hasattr(obj, attr_name):
        return True

    # Check accessors (rst, vct, pc)
    for accessor_name in accessors:
        accessor = getattr(obj, accessor_name, None)
        if accessor is not None and hasattr(accessor, attr_name):
            return True

    return False


