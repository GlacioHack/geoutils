"""Test for PointCloud class."""
from __future__ import annotations

import pytest
import numpy as np
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal

from geoutils import PointCloud


class TestPointCloud:

    # 1/ Synthetic point cloud with no auxiliary column
    rng = np.random.default_rng(42)
    points = rng.integers(low=1, high=1000, size=(100, 2)) + rng.normal(0, 0.15, size=(100, 2))
    val1 = rng.normal(scale=3, size=100)
    pc1 = gpd.GeoDataFrame(data={"b1": val1}, geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]), crs=4326)

    # 2/ LAS file
    filename = "/home/atom/ongoing/own/geoutils/points.laz"

    def test_init(self):
        """Test instantiation of a point cloud."""

        # 1/ For a single column point cloud
        pc = PointCloud(self.pc1, data_column="b1")

        # Assert that both the dataframe and data column name are equal
        assert pc.data_column == "b1"
        assert_geodataframe_equal(pc.ds, self.pc1)

        # 2/ For a point cloud from filie
        pc = PointCloud(self.filename, data_column="Z")

        assert pc.data_column == "Z"
        assert not pc.is_loaded


    def test_data_column_name(self):
        pass

    def test_from_array(self):
        """Test building point cloud from array."""

    def test_from_tuples(self):
        pass

    def test_to_array(self):
        pass

    def test_to_tuples(self):
        pass