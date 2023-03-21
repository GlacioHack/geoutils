"""
Test functions for SatelliteImage class
"""
import datetime
import datetime as dt
import sys
from io import StringIO

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples

DO_PLOT = False


class TestSatelliteImage:
    landsat_b4 = examples.get_path("everest_landsat_b4")
    aster_dem = examples.get_path("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4, aster_dem])  # type: ignore
    def test_init(self, example: str) -> None:
        """
        Test that inputs work properly in SatelliteImage class init
        """

        # from filename, checking option
        img = gu.SatelliteImage(example, read_from_fn=False)
        img = gu.SatelliteImage(example)
        assert isinstance(img, gu.SatelliteImage)

        # from SatelliteImage
        img2 = gu.SatelliteImage(img)
        assert isinstance(img2, gu.SatelliteImage)

        # from Raster
        r = gu.Raster(example)
        img3 = gu.SatelliteImage(r)
        assert isinstance(img3, gu.SatelliteImage)

        assert img.raster_equal(img2)
        assert img.raster_equal(img3)

    @pytest.mark.parametrize("example", [landsat_b4, aster_dem])  # type: ignore
    def test_silent(self, example: str) -> None:
        """
        Test that the silent method does not return any output in console
        """

        # let's capture stdout
        # cf https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
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
            gu.SatelliteImage(example, silent=False)

        # check the metadata reading outputs to console
        assert len(output1) > 0

        with Capturing() as output2:
            gu.SatelliteImage(example, silent=True)

        # check nothing outputs to console
        assert len(output2) == 0

    def test_add_sub(self) -> None:
        """
        Test that overloading of addition, subtraction and negation works for child classes as well.
        """
        # Create fake rasters with random values in 0-255 and dtype uint8
        width = height = 5
        transform = rio.transform.from_bounds(0, 0, 1, 1, width, height)
        satimg1 = gu.SatelliteImage.from_array(
            np.random.randint(0, 255, (height, width), dtype="uint8"), transform=transform, crs=None
        )
        satimg2 = gu.SatelliteImage.from_array(
            np.random.randint(0, 255, (height, width), dtype="uint8"), transform=transform, crs=None
        )

        # Check that output type is same - other tests are in test_raster.py
        sat_out = -satimg1
        assert isinstance(sat_out, gu.SatelliteImage)

        sat_out = satimg1 + satimg2
        assert isinstance(sat_out, gu.SatelliteImage)

        sat_out = satimg1 - satimg2  # type: ignore
        assert isinstance(sat_out, gu.SatelliteImage)

    @pytest.mark.parametrize("example", [landsat_b4, aster_dem])  # type: ignore
    def test_copy(self, example: str) -> None:
        """
        Test that the copy method works as expected for SatelliteImage. In particular
        when copying r to r2:
        - if r.data is modified and r copied, the updated data is copied
        - if r is copied, r.data changed, r2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        r = gu.SatelliteImage(example)
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a SatelliteImage
        assert isinstance(r2, gu.SatelliteImage)

        # check all immutable attributes are equal
        raster_attrs = [
            "bounds",
            "count",
            "crs",
            "dtypes",
            "height",
            "indexes",
            "nodata",
            "res",
            "shape",
            "transform",
            "width",
        ]
        satimg_attrs = ["satellite", "sensor", "product", "version", "tile_name", "datetime"]
        # using list directly available in Class
        attrs = raster_attrs + satimg_attrs
        all_attrs = attrs + gu.raster.satimg.satimg_attrs
        for attr in all_attrs:
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert np.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.array_equal(r.data.mask, r2.data.mask)

        # Check that if r.data is modified, it does not affect r2.data
        r.data += 5
        assert not np.array_equal(r.data, r2.data, equal_nan=True)

    def test_filename_parsing(self) -> None:
        """Test metadata parsing from filenames"""

        copied_names = [
            "TDM1_DEM__30_N00E104_DEM.tif",
            "SETSM_WV02_20141026_1030010037D17F00_10300100380B4000_mosaic5_2m_v3.0_dem.tif",
            "SETSM_s2s041_WV02_20150615_10300100443C2D00_1030010043373000_seg1_2m_dem.tif",
            "AST_L1A_00303132015224418_final.tif",
            "ILAKS1B_20190928_271_Gilkey-DEM.tif",
            "srtm_06_01.tif",
            "ASTGTM2_N00E108_dem.tif",
            "N00E015.hgt",
            "NASADEM_HGT_n00e041.hgt",
        ]
        # Corresponding data, filled manually
        satellites = ["TanDEM-X", "WorldView", "WorldView", "Terra", "IceBridge", "SRTM", "Terra", "SRTM", "SRTM"]
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
        # we can skip the version, bit subjective...
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
            assert satellites[i] == attrs[0]
            assert sensors[i] == attrs[1]
            assert products[i] == attrs[2]
            assert tiles[i] == attrs[4]
            assert datetimes[i] == attrs[5]

    def test_sw_tile_naming_parsing(self) -> None:
        # normal examples
        test_tiles = ["N14W065", "S14E065", "N014W065", "W065N014", "W065N14", "N00E000"]
        test_latlon = [(14, -65), (-14, 65), (14, -65), (14, -65), (14, -65), (0, 0)]

        for tile in test_tiles:
            assert gu.raster.satimg.sw_naming_to_latlon(tile)[0] == test_latlon[test_tiles.index(tile)][0]
            assert gu.raster.satimg.sw_naming_to_latlon(tile)[1] == test_latlon[test_tiles.index(tile)][1]

        for latlon in test_latlon:
            assert gu.raster.satimg.latlon_to_sw_naming(latlon) == test_tiles[test_latlon.index(latlon)]

        # check possible exceptions, rounded lat/lon belong to their southwest border
        assert gu.raster.satimg.latlon_to_sw_naming((0, 0)) == "N00E000"
        # those are the same point, should give same naming
        assert gu.raster.satimg.latlon_to_sw_naming((-90, 0)) == "S90E000"
        assert gu.raster.satimg.latlon_to_sw_naming((90, 0)) == "S90E000"
        # same here
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

        # Landsat 1
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
