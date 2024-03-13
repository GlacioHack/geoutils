"""Setup of runtime-compile configuration of GeoUtils."""
import configparser
import os
from typing import Any

# The setup is inspired by that of Matplotlib and Geowombat
# https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/rcsetup.py
# https://github.com/jgrss/geowombat/blob/main/src/geowombat/config.py

_config_ini_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.ini"))

# Validators: to check the format of user inputs
def validate_bool(b):
    """Convert b to ``bool`` or raise."""
    if isinstance(b, str):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError(f'Cannot convert {b!r} to bool')


# Map the parameter names with a validating function to check user input
_validators = {
    "shift_area_or_point": validate_bool,
    "warn_area_or_point":  validate_bool,
}


class GeoUtilsConfigDict(dict):
    """Class for a GeoUtils config dictionary"""

    def __setitem__(self, k: str, v: Any):
        """We override setitem to check user input."""

        validate_func = _validators[k]
        new_value = validate_func(v)
        super().__setitem__(k, new_value)

    def _set_defaults(self, path_init_file: str):
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

