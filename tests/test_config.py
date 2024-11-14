"""Test configuration file."""

import geoutils as gu


class TestConfig:
    def test_config_defaults(self) -> None:
        """Check defaults compared to file"""

        # Read file
        default_config = gu._config.GeoUtilsConfigDict()
        default_config._set_defaults(gu._config._config_ini_file)

        assert default_config == gu.config

    def test_config_set(self) -> None:
        """Check setting a non-default config argument by user"""

        # Default is True
        assert gu.config["shift_area_or_point"]

        # We set it to False and it should be updated
        gu.config["shift_area_or_point"] = False
        assert not gu.config["shift_area_or_point"]

        # Leave the test with the initial default
        gu.config["shift_area_or_point"] = True
        assert gu.config["shift_area_or_point"]

    def test_config_validator(self) -> None:
        """Check setting a config argument with a wrong input type converts it automatically"""

        # We input an "off" value, that should be converted to False
        gu.config["shift_area_or_point"] = "off"
        assert not gu.config["shift_area_or_point"]

        # Leave the test with initial default
        gu.config["shift_area_or_point"] = 1
        assert gu.config["shift_area_or_point"]
