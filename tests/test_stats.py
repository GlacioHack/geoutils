"""
Test functions for stats
"""

import scipy

from geoutils import Raster, examples
from geoutils.stats import nmad


class TestStats:
    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_raster = Raster(landsat_b4_path)

    def test_nmad(self) -> None:
        """Test NMAD functionality runs on any type of input"""

        # Check that the NMAD is computed the same with a masked array or NaN array, and is equal to scipy nmad
        nmad_ma = nmad(self.landsat_raster.data)
        nmad_array = nmad(self.landsat_raster.get_nanarray())
        nmad_scipy = scipy.stats.median_abs_deviation(self.landsat_raster.data, axis=None, scale="normal")

        assert nmad_ma == nmad_array
        assert nmad_ma.round(2) == nmad_scipy.round(2)

        # Check that the scaling factor works
        nmad_1 = nmad(self.landsat_raster.data, nfact=1)
        nmad_2 = nmad(self.landsat_raster.data, nfact=2)

        assert nmad_1 * 2 == nmad_2
