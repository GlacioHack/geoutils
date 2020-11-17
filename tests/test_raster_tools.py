from unittest import TestCase

import rastertools

class TestReading(TestCase):

	def test_open_file(self):
		im = rastertools.Raster(test_image)

class TestInfo(TestCase):
	
    def test_is_string(self):
        s = rastertools.info()
        self.assertTrue(isinstance(s, basestring))