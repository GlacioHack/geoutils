"""Test tiling tools for arrays and rasters."""

import numpy as np
import pytest

import geoutils as gu
from geoutils import examples


class TestTiling:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")

    def test_subdivide_array(self) -> None:
        # Import optional scikit-image or skip test

        test_shape = (6, 4)
        test_count = 4
        subdivision_grid = gu.raster.subdivide_array(test_shape, test_count)

        assert subdivision_grid.shape == test_shape
        assert np.unique(subdivision_grid).size == test_count

        assert np.unique(gu.raster.subdivide_array((3, 3), 3)).size == 3

        with pytest.raises(ValueError, match=r"Expected a 2D shape, got 1D shape.*"):
            gu.raster.subdivide_array((5,), 2)

        with pytest.raises(ValueError, match=r"Shape.*smaller than.*"):
            gu.raster.subdivide_array((5, 2), 15)
