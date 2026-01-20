"""Tests on Xarray accessor mirroring Raster API."""

import warnings

import pytest
import numpy as np

from geoutils import examples, open_raster

class TestAccessor:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_open_raster(self):
        pass

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_copy(self, path_raster: str):

        ds = open_raster(path_raster)
        ds_copy = ds.rst.copy()

        assert np.array_equal(ds.data, ds_copy.data, equal_nan=True)
        assert ds.rst.transform == ds_copy.rst.transform
        assert ds.rst.crs == ds_copy.rst.crs
        assert ds.rst.nodata == ds_copy.rst.nodata