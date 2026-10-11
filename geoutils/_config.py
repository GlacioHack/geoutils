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

"""Setup of runtime-compile configuration of GeoUtils."""

from __future__ import annotations

import configparser
import os
from typing import Any

from rasterio.enums import Resampling

# The setup is inspired by that of Matplotlib and Geowombat
# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/rcsetup.py
# https://github.com/jgrss/geowombat/blob/main/src/geowombat/config.py

_config_ini_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.ini"))

# Validators: to check the format of user inputs


def validate_bool(b: bool | str | int) -> bool:
    """Convert b to ``bool`` or raise."""
    if isinstance(b, str):
        b = b.lower()
    if b in ("t", "y", "yes", "on", "true", "1", 1, True):
        return True
    elif b in ("f", "n", "no", "off", "false", "0", 0, False):
        return False
    else:
        raise ValueError(f"Cannot convert {b!r} to bool")


def validate_reprojection_method(reprojection_method: bool | str | int) -> str:
    """Test reprojection_method"""
    if isinstance(reprojection_method, str):
        if reprojection_method.lower() in [method.name for method in Resampling]:
            return reprojection_method.lower()
    raise ValueError(
        f"'{reprojection_method}' is not a valid rasterio.enums.Resampling method"
        f"Valid methods: {[method.name for method in Resampling]}"
    )


def validate_interpolation_method(interpolation_method: bool | str | int) -> str:
    """Test interpolation_method"""
    valid_methods = ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    if isinstance(interpolation_method, str) and interpolation_method.lower() in valid_methods:
        return interpolation_method.lower()
    else:
        raise ValueError(f"'{interpolation_method}' is not a valid interpolation methodValid methods: {valid_methods}")


def validate_nodata_handling(nodata_handling: bool | str | int) -> str | int:
    """Validate how interpolation handles missing source values."""
    valid_choices = ["nearest", "ignore", "propagate", "half_order_up", "half_order_down"]
    if isinstance(nodata_handling, str):
        if nodata_handling.lower() in valid_choices:
            return nodata_handling.lower()
        try:
            configured_distance = int(nodata_handling)
        except ValueError:
            configured_distance = -1
        if configured_distance >= 0:
            return configured_distance
    elif not isinstance(nodata_handling, bool) and isinstance(nodata_handling, int) and nodata_handling >= 0:
        return nodata_handling
    raise ValueError(
        f"'{nodata_handling}' is not a valid nodata_handling choice. Use {valid_choices} or a non-negative integer."
    )


# Map the parameter names with a validating function to check user input
_validators = {
    "shift_area_or_point": validate_bool,
    "warn_area_or_point": validate_bool,
    "reprojection_method": validate_reprojection_method,
    "interpolation_method": validate_interpolation_method,
    "interpolation_nodata_handling": validate_nodata_handling,
}


class GeoUtilsConfigDict(dict):  # type: ignore
    """Class for a GeoUtils config dictionary"""

    def __setitem__(self, k: str, v: Any) -> None:
        """We override setitem to check user input."""

        validate_func = _validators[k]
        new_value = validate_func(v)
        super().__setitem__(k, new_value)

    def _set_defaults(self, path_init_file: str) -> None:
        """A function to set"""

        config_parser = configparser.ConfigParser()
        config_parser.read(path_init_file)

        for section in config_parser.sections():
            for k, v in config_parser[section].items():
                # Select validator function and update dictionary
                validate_func = _validators[k]
                self.__setitem__(k, validate_func(v))


# Generate default config dictionary
config = GeoUtilsConfigDict()
config._set_defaults(path_init_file=_config_ini_file)
