"""
Test functions for satimg
"""
import os
import inspect
import geoutils.georaster as gr
import geoutils.satimg as si
import pytest


DO_PLOT = False

@pytest.fixture()
def path_data():
    path_module = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getsourcefile(gr))))
    fn_img = os.path.join(path_module, 'tests', 'data', 'LE71400412000304SGS00_B4_crop.TIF')
    fn_img2 = os.path.join(path_module, 'tests', 'data', 'LE71400412000304SGS00_B4_crop2.TIF')

    return fn_img, fn_img2

class TestSatelliteImage:

    def test_load_subclass(self,path_data):

        fn_img, _ = path_data

        img = si.SatelliteImage(fn_img,read_from_fn=False)


    def test_filename_parsing(self):

        copied_names = ['TDM1_DEM__30_N00E104_DEM.tif',
                        'SETSM_WV02_20141026_1030010037D17F00_10300100380B4000_mosaic5_2m_v3.0_dem.tif',
                        'AST_L1A_00303132015224418_final.tif',
                        'ILAKS1B_20190928_271_Gilkey-DEM.tif',
                        'srtm_06_01.tif',
                        'ASTGTM2_N00E108_dem.tif',
                        'N00E015.hgt',
                        'NASADEM_HGT_n00e041.hgt']

        for names in copied_names:
            attrs = si.parse_metadata_from_fn(names)
            print(attrs)

    def test_tile_ext_parsing(self):

        pass



