# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Define shared ways to handle invalid values during interpolation and gridding."""

from __future__ import annotations

from typing import Literal, cast

NodataPropagation = Literal["gdal", "ignore", "propagate"]


def _validate_nodata_propagation(nodata_propagation: str) -> NodataPropagation:
    """
    Validate and normalize a nodata propagation rule.

    :param nodata_propagation: Rule used to handle invalid source values.

    :return: Normalized propagation rule.
    """

    # Lowercase string values so public methods accept the same spelling variants
    normalized = nodata_propagation.lower()
    if normalized not in ("gdal", "ignore", "propagate"):
        raise ValueError("nodata_propagation must be one of 'gdal', 'ignore' or 'propagate'.")
    return cast(NodataPropagation, normalized)
