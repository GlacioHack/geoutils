"""
Test projtools
"""
import numpy as np

import geoutils.georaster as gr
import geoutils.projtools as pt
from geoutils import datasets


class TestProjTools:

    def test_latlon_reproject(self):
        """
        Check that to and from latlon projections are self consistent within tolerated rounding errors
        """

        img = gr.Raster(datasets.get_path('landsat_B4'))

        # Test on random points
        nsample = 100
        randx = np.random.randint(low=img.bounds.left, high=img.bounds.right, size=(nsample,))
        randy = np.random.randint(low=img.bounds.bottom, high=img.bounds.top, size=(nsample,))

        lat, lon = pt.reproject_to_latlon([randx, randy], img.crs)
        x, y = pt.reproject_from_latlon([lat, lon], img.crs)

        assert np.all(x == randx)
        assert np.all(y == randy)
