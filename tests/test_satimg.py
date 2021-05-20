"""
Test functions for SatelliteImage class
"""
import os
import sys
import pytest
import datetime as dt
import numpy as np
import geoutils.georaster as gr
import geoutils.satimg as si
from geoutils import datasets
import geoutils
from io import StringIO
import numpy as np

DO_PLOT = False


class TestSatelliteImage:

    def test_init(self):
        """
        Test that inputs work properly in SatelliteImage class init
        """

        fn_img = datasets.get_path("landsat_B4")

        # from filename, checking option
        img = si.SatelliteImage(fn_img, read_from_fn=False)
        img = si.SatelliteImage(fn_img)
        assert isinstance(img, si.SatelliteImage)

        # from SatelliteImage
        img2 = si.SatelliteImage(img)
        assert isinstance(img2, si.SatelliteImage)

        # from Raster
        r = gr.Raster(fn_img)
        img3 = si.SatelliteImage(r)
        assert isinstance(img3, si.SatelliteImage)

        assert np.logical_and.reduce((np.array_equal(img.data, img2.data, equal_nan=True),
                                      np.array_equal(img2.data, img3.data, equal_nan=True)))

        assert np.logical_and.reduce((np.all(img.data.mask == img2.data.mask),
                                      np.all(img2.data.mask == img3.data.mask)))

    def test_silent(self):
        """
        Test that the silent method does not return any output in console
        """
        fn_img = datasets.get_path("landsat_B4")

        # let's capture stdout
        # cf https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
        class Capturing(list):
            def __enter__(self):
                self._stdout = sys.stdout
                sys.stdout = self._stringio = StringIO()
                return self

            def __exit__(self, *args):
                self.extend(self._stringio.getvalue().splitlines())
                del self._stringio  # free up some memory
                sys.stdout = self._stdout

        with Capturing() as output1:
            img = si.SatelliteImage(fn_img)

        # check the metadata reading outputs to console
        assert len(output1) > 0

        with Capturing() as output2:
            img = si.SatelliteImage(fn_img, silent=True)

        # check nothing outputs to console
        assert len(output2) == 0

    def test_copy(self):
        """
        Test that the copy method works as expected for SatelliteImage. In particular
        when copying r to r2:
        - if r.data is modified and r copied, the updated data is copied
        - if r is copied, r.data changed, r2.data should be unchanged
        """
        # Open dataset, update data and make a copy
        r = si.SatelliteImage(datasets.get_path("landsat_B4"))
        r.data += 5
        r2 = r.copy()

        # Objects should be different (not pointing to the same memory)
        assert r is not r2

        # Check the object is a SatelliteImage
        assert isinstance(r2, geoutils.satimg.SatelliteImage)

        # check all immutable attributes are equal
        # georaster_attrs = ['bounds', 'count', 'crs', 'dtypes', 'height', 'indexes', 'nodata',
        #                    'res', 'shape', 'transform', 'width']
        # satimg_attrs = ['satellite', 'sensor', 'product', 'version', 'tile_name', 'datetime']
        # using list directly available in Class
        attrs = [at for at in r._get_rio_attrs() if at not in ['name', 'dataset_mask', 'driver']]
        all_attrs = attrs + si.satimg_attrs
        for attr in all_attrs:
            assert r.__getattribute__(attr) == r2.__getattribute__(attr)

        # Check data array
        assert np.array_equal(r.data, r2.data, equal_nan=True)

        # Check dataset_mask array
        assert np.all(r.data.mask == r2.data.mask)

        # Check that if r.data is modified, it does not affect r2.data
        r.data += 5
        assert not np.array_equal(r.data, r2.data, equal_nan=True)

    def test_filename_parsing(self):

        copied_names = ['TDM1_DEM__30_N00E104_DEM.tif',
                        'SETSM_WV02_20141026_1030010037D17F00_10300100380B4000_mosaic5_2m_v3.0_dem.tif',
                        'AST_L1A_00303132015224418_final.tif',
                        'ILAKS1B_20190928_271_Gilkey-DEM.tif',
                        'srtm_06_01.tif',
                        'ASTGTM2_N00E108_dem.tif',
                        'N00E015.hgt',
                        'NASADEM_HGT_n00e041.hgt']
        # Corresponding data, filled manually
        satellites = ['TanDEM-X', 'WorldView', 'Terra', 'IceBridge', 'SRTM',
                      'Terra', 'SRTM', 'SRTM']
        sensors = ['TanDEM-X', 'WV02', 'ASTER', 'UAF-LS', 'SRTM', 'ASTER',
                   'SRTM', 'SRTM']
        products = ['TDM1', 'ArcticDEM/REMA', 'L1A', 'ILAKS1B', 'SRTMv4.1',
                    'ASTGTM2', 'SRTMGL1', 'NASADEM-HGT']
        # we can skip the version, bit subjective...
        tiles = ['N00E104', None, None, None, '06_01', 'N00E108', 'N00E015',
                 'n00e041']
        datetimes = [None, dt.datetime(year=2014, month=10, day=26), dt.datetime(year=2015, month=3, day=13, hour=22, minute=44, second=18),
                     dt.datetime(year=2019, month=9, day=28), dt.datetime(year=2000, month=2,
                                                                          day=15), None, dt.datetime(year=2000, month=2, day=15),
                     dt.datetime(year=2000, month=2, day=15)]

        for names in copied_names:
            attrs = si.parse_metadata_from_fn(names)
            i = copied_names.index(names)
            assert satellites[i] == attrs[0]
            assert sensors[i] == attrs[1]
            assert products[i] == attrs[2]
            assert tiles[i] == attrs[4]
            assert datetimes[i] == attrs[5]

    def test_sw_tile_naming_parsing(self):

        # normal examples
        test_tiles = ['N14W065', 'S14E065', 'N014W065', 'W065N014', 'W065N14', 'N00E000']
        test_latlon = [(14, -65), (-14, 65), (14, -65), (14, -65), (14, -65), (0, 0)]

        for tile in test_tiles:
            assert si.sw_naming_to_latlon(tile)[0] == test_latlon[test_tiles.index(tile)][0]
            assert si.sw_naming_to_latlon(tile)[1] == test_latlon[test_tiles.index(tile)][1]

        for latlon in test_latlon:
            assert si.latlon_to_sw_naming(latlon) == test_tiles[test_latlon.index(latlon)]

        # check possible exceptions, rounded lat/lon belong to their southwest border
        assert si.latlon_to_sw_naming((0, 0)) == 'N00E000'
        # those are the same point, should give same naming
        assert si.latlon_to_sw_naming((-90, 0)) == 'S90E000'
        assert si.latlon_to_sw_naming((90, 0)) == 'S90E000'
        # same here
        assert si.latlon_to_sw_naming((0, -180)) == 'N00W180'
        assert si.latlon_to_sw_naming((0, 180)) == 'N00W180'

