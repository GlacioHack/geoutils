"""
Test functions for geoutils
"""
import os
import unittest
import rasterio as rio

import geoutils.georaster as gr
import geoutils.geovector as gv

test_data_path = os.path.join('/', *(__file__.split('/')[:-2]), 'tests', 'data')
test_img = os.path.join(test_data_path,'LE71400412000304SGS00_B4_crop.TIF')

# class TestReadingRaster(unittest.TestCase):
#
# 	def test_open_file(self):
# 		im = gr.Raster(test_img)
#         self.assertTrue(isinstance(im,rio.io.MemoryFile))
#
#     def test_is_string(self):
#         im = gr.Raster(test_img)
#         s = im.info()
#         self.assertTrue(isinstance(s, str))