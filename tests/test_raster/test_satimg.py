"""
Test functions for metadata parsing from sensor, often satellite imagery.
"""

from __future__ import annotations

import datetime
import datetime as dt
import os
import sys
import tempfile
from io import StringIO

import pytest

import geoutils as gu
from geoutils import examples
from geoutils.raster.satimg import satimg_tags

DO_PLOT = False


class TestSatImg:

    landsat_b4 = examples.get_path_test("everest_landsat_b4")
    aster_dem = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4, aster_dem])  # type: ignore
    def test_init(self, example: str) -> None:
        """Test that the sensor reading through Raster initialisation works."""

        # Load with parse sensor metadata, it should write metadata in the tags
        rast = gu.Raster(example, parse_sensor_metadata=True)
        for tag in satimg_tags:
            assert tag in rast.tags.keys()

        # And that otherwise it does not
        rast = gu.Raster(example, parse_sensor_metadata=False)
        for tag in satimg_tags:
            assert tag not in rast.tags.keys()

    @pytest.mark.parametrize("example", [landsat_b4, aster_dem])  # type: ignore
    def test_silent(self, example: str) -> None:
        """
        Test that the silent method does not return any output in console
        """

        # Let's capture stdout
        # See https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
        class Capturing(list):  # type: ignore
            def __enter__(self):  # type: ignore
                self._stdout = sys.stdout
                sys.stdout = self._stringio = StringIO()
                return self

            def __exit__(self, *args) -> None:  # type: ignore
                self.extend(self._stringio.getvalue().splitlines())
                del self._stringio  # free up some memory
                sys.stdout = self._stdout

        with Capturing() as output1:
            gu.Raster(example, parse_sensor_metadata=True, silent=False)

        # Check the metadata reading outputs to console
        assert len(output1) > 0

        with Capturing() as output2:
            gu.Raster(example, parse_sensor_metadata=True, silent=True)

        # Check nothing outputs to console
        assert len(output2) == 0

    @pytest.mark.parametrize("example", [landsat_b4, aster_dem])  # type: ignore
    def test_save_tags(self, example: str) -> None:
        """Check that the metadata read is saved in tags of raster metadata."""

        rast = gu.Raster(example, parse_sensor_metadata=True)

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test.tif")
        rast.to_file(temp_file)
        saved = gu.Raster(temp_file)
        saved_tags = saved.tags
        rast_tags = rast.tags
        # Do not check COMPRESSION tags
        # The file "saved" is just read, it has no compression by default (no COMPRESSION tag)
        # The file "rast" is saved, therefore a compression is applied (COMPRESSION tag)
        saved_tags.pop("COMPRESSION", None)
        rast_tags.pop("COMPRESSION", None)
        assert saved_tags == rast_tags

    def test_filename_parsing(self) -> None:
        """Test metadata parsing from filenames"""

        copied_names = [
            "TDM1_DEM__30_N00E104_DEM.tif",
            "SETSM_WV02_20141026_ex1030010037D17F00_10300100380B4000_mosaic5_2m_v3.0_dem.tif",
            "SETSM_s2s041_WV02_20150615_10300100443C2D00_1030010043373000_seg1_2m_dem.tif",
            "AST_L1A_00303132015224418_final.tif",
            "ILAKS1B_20190928_271_Gilkey-DEM.tif",
            "srtm_06_01.tif",
            "ASTGTM2_N00E108_dem.tif",
            "N00E015.hgt",
            "NASADEM_HGT_n00e041.hgt",
        ]
        # Corresponding data, filled manually
        platform = ["TanDEM-X", "WorldView", "WorldView", "Terra", "IceBridge", "SRTM", "Terra", "SRTM", "SRTM"]
        sensors = ["TanDEM-X", "WV02", "WV02", "ASTER", "UAF-LS", "SRTM", "ASTER", "SRTM", "SRTM"]
        products = [
            "TDM1",
            "ArcticDEM/REMA/EarthDEM",
            "ArcticDEM/REMA/EarthDEM",
            "L1A",
            "ILAKS1B",
            "SRTMv4.1",
            "ASTGTM2",
            "SRTMGL1",
            "NASADEM-HGT",
        ]
        # We can skip the version, bit subjective...
        tiles = ["N00E104", None, None, None, None, "06_01", "N00E108", "N00E015", "n00e041"]
        datetimes = [
            None,
            dt.datetime(year=2014, month=10, day=26),
            dt.datetime(year=2015, month=6, day=15),
            dt.datetime(year=2015, month=3, day=13, hour=22, minute=44, second=18),
            dt.datetime(year=2019, month=9, day=28),
            dt.datetime(year=2000, month=2, day=15),
            None,
            dt.datetime(year=2000, month=2, day=15),
            dt.datetime(year=2000, month=2, day=15),
        ]

        for names in copied_names:
            attrs = gu.raster.satimg.parse_metadata_from_fn(names)
            i = copied_names.index(names)
            assert platform[i] == attrs["platform"]
            assert sensors[i] == attrs["sensor"]
            assert products[i] == attrs["product"]
            assert tiles[i] == attrs["tile_name"]
            assert datetimes[i] == attrs["datetime"]

    def test_sw_tile_naming_parsing(self) -> None:
        # normal examples
        test_tiles = ["N14W065", "S14E065", "N014W065", "W065N014", "W065N14", "N00E000"]
        test_latlon = [(14, -65), (-14, 65), (14, -65), (14, -65), (14, -65), (0, 0)]

        for tile in test_tiles:
            assert gu.raster.satimg.sw_naming_to_latlon(tile)[0] == test_latlon[test_tiles.index(tile)][0]
            assert gu.raster.satimg.sw_naming_to_latlon(tile)[1] == test_latlon[test_tiles.index(tile)][1]

        for latlon in test_latlon:
            assert gu.raster.satimg.latlon_to_sw_naming(latlon) == test_tiles[test_latlon.index(latlon)]

        # Check possible exceptions, rounded lat/lon belong to their southwest border
        assert gu.raster.satimg.latlon_to_sw_naming((0, 0)) == "N00E000"
        # Those are the same point, should give same naming
        assert gu.raster.satimg.latlon_to_sw_naming((-90, 0)) == "S90E000"
        assert gu.raster.satimg.latlon_to_sw_naming((90, 0)) == "S90E000"
        # Same here
        assert gu.raster.satimg.latlon_to_sw_naming((0, -180)) == "N00W180"
        assert gu.raster.satimg.latlon_to_sw_naming((0, 180)) == "N00W180"

    def test_parse_tile_attr_from_name(self) -> None:
        """Test the parsing of tile attribute from tile name."""

        # For ASTER, SRTM, NASADEM: 1x1 tiling globally
        y, x, size, epsg = gu.raster.satimg.parse_tile_attr_from_name(tile_name="N01W179", product="SRTMGL1")

        assert y == 1
        assert x == -179
        assert size == (1, 1)
        assert epsg == 4326

        # For TanDEM-X: depends on latitude
        # Mid-latitude is 2 x 1
        y, x, size, epsg = gu.raster.satimg.parse_tile_attr_from_name(tile_name="N62E04", product="TDM1")
        assert y == 62
        assert x == 4
        assert size == (1, 2)
        assert epsg == 4326

        # Low-latitude is 1 x 1
        y, x, size, epsg = gu.raster.satimg.parse_tile_attr_from_name(tile_name="N52E04", product="TDM1")
        assert y == 52
        assert x == 4
        assert size == (1, 1)
        assert epsg == 4326

        # High-latitude is 5 x 1
        y, x, size, epsg = gu.raster.satimg.parse_tile_attr_from_name(tile_name="N82E04", product="TDM1")
        assert y == 82
        assert x == 4
        assert size == (1, 4)
        assert epsg == 4326

    def test_parse_landsat(self) -> None:
        """Test the parsing of landsat metadata from name."""

        # Landsat 1 example
        landsat1 = "LM10170391976031AAA01.tif"
        attrs1 = gu.raster.satimg.parse_landsat(landsat1)

        assert attrs1[0] == "Landsat 1"
        assert attrs1[1] == "MSS"
        assert attrs1[-1] == datetime.datetime(1976, 1, 31)

        # Landsat 7 example
        landsat7 = "LE71400412000304SGS00_B4.tif"
        attrs7 = gu.raster.satimg.parse_landsat(landsat7)

        assert attrs7[0] == "Landsat 7"
        assert attrs7[1] == "ETM+"
        assert attrs7[-1] == datetime.datetime(2000, 10, 30)
