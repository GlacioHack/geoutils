"""
Test functions for stats
"""

import numpy as np
import scipy

from geoutils import Raster, examples
from geoutils.stats import linear_error, nmad


class TestStats:
    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_raster = Raster(landsat_b4_path)

    def test_nmad(self) -> None:
        """Test NMAD functionality runs on any type of input"""

        # Check that the NMAD is computed the same with a masked array or NaN array, and is equal to scipy nmad
        nmad_ma = nmad(self.landsat_raster.data)
        nmad_array = nmad(self.landsat_raster.get_nanarray(floating_dtype="float64"))
        nmad_scipy = scipy.stats.median_abs_deviation(self.landsat_raster.data, axis=None, scale="normal")

        assert nmad_ma == nmad_array
        assert nmad_ma.round(2) == nmad_scipy.round(2)

        # Check that the scaling factor works
        nmad_1 = nmad(self.landsat_raster.data, nfact=1)
        nmad_2 = nmad(self.landsat_raster.data, nfact=2)

        assert nmad_1 * 2 == nmad_2

    def test_linear_error(self) -> None:
        """Test linear error (LE) functionality runs on any type of input"""

        # Compute LE on the landsat raster data for a default interval (LE90)
        le_ma = linear_error(self.landsat_raster.data)
        le_nan_array = linear_error(self.landsat_raster.get_nanarray())

        # Assert the LE90 is computed the same for masked array and NaN array
        assert le_ma == le_nan_array

        # Check that the function works for different intervals
        le90 = linear_error(self.landsat_raster.data, interval=90)
        le50 = linear_error(self.landsat_raster.data, interval=50)

        # Verify that LE50 (interquartile range) is smaller than LE90
        assert le50 < le90

        # Test a known dataset
        test_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        le90_test = linear_error(test_data, interval=90)
        le50_test = linear_error(test_data, interval=50)

        assert le90_test == 9
        assert le50_test == 5

        # Test masked arrays with invalid data (should ignore NaNs/masked values)
        masked_data = np.ma.masked_array(test_data, mask=[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        le90_masked = linear_error(masked_data, interval=90)
        le50_masked = linear_error(masked_data, interval=50)

        assert le90_masked == 4.5
        assert le50_masked == 2.5
