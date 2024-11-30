"""Test for PointCloud class."""
from __future__ import annotations

import pytest
from pyproj import CRS
from rasterio.coords import BoundingBox
import numpy as np
import geopandas as gpd
from shapely import Polygon
from geopandas.testing import assert_geodataframe_equal

from geoutils import PointCloud


class TestPointCloud:

    # 1/ Synthetic point cloud with no auxiliary column
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(100, 3)) + rng.normal(0, 0.15, size=(100, 3))
    gdf1 = gpd.GeoDataFrame(data={"b1": arr_points[:, 2]},
                            geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1]), crs=4326)

    # 2/ Synthetic point cloud with auxiliary column
    arr_points2 = rng.integers(low=1, high=1000, size=(100, 4)) + rng.normal(0, 0.15, size=(100, 4))
    gdf2 = gpd.GeoDataFrame(data=arr_points[:, 2:], columns=["b1", "b2"],
                            geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1]), crs=4326)
    # 2/ LAS file
    fn_las = "/home/atom/ongoing/own/geoutils/points.laz"

    # 3/ Non-point vector (for error raising)
    poly = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    gdf3 = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

    def test_init(self) -> None:
        """Test instantiation of a point cloud."""

        # 1/ For a single column point cloud
        pc = PointCloud(self.gdf1, data_column="b1")

        # Assert that both the dataframe and data column name are equal
        assert pc.data_column == "b1"
        assert_geodataframe_equal(pc.ds, self.gdf1)

        # 2/ For a point cloud from LAS/LAZ file
        pc = PointCloud(self.fn_las, data_column="Z")

        assert pc.data_column == "Z"
        assert not pc.is_loaded

    def test_init__errors(self) -> None:
        """Test errors raised by point cloud instantiation."""

        # If the data column does not exist
        with pytest.raises(ValueError, match="Data column column_that_does_not_exist not found*"):
            PointCloud(self.gdf1, data_column="column_that_does_not_exist")

        # If vector is not only comprised of points
        with pytest.raises(ValueError, match="This vector file contains non-point geometries*"):
            PointCloud(self.gdf3)

    def test_load(self) -> None:
        """
        Test loading of a point cloud (only possible with a LAS file).

        This test also serves to test the overridden methods "crs", "bounds", "nb_points", "all_columns" in relation
        to loading.
        """

        # 1/ Check unloaded and loaded attributes are all the same

        pc = PointCloud(self.fn_las, data_column="Z")

        # Check point cloud is not loaded, and fetch before metadata
        assert not pc.is_loaded
        before_crs = pc.crs
        before_bounds = pc.bounds
        before_nb_points = pc.nb_points
        before_columns = pc.all_columns

        # Load and fetch after metadata
        pc.load(columns="all")
        assert pc.is_loaded

        after_crs = pc.crs
        after_bounds = pc.bounds
        after_nb_points = pc.nb_points
        after_columns = pc.all_columns

        # Check those are equal
        assert before_crs == after_crs
        assert before_bounds == after_bounds
        assert before_nb_points == after_nb_points
        assert all(before_columns == after_columns)

        # 2/ Check default column argument
        pc = PointCloud(self.fn_las, data_column="Z")
        pc.load()

        assert pc.all_columns == ["Z"]

        # 3/ Check implicit loading when calling a function requiring .ds
        pc = PointCloud(self.fn_las, data_column="Z")
        assert not pc.is_loaded

        pc2 = pc.buffer(distance=0.1)
        assert pc.is_loaded

    def test_load__errors(self) -> None:
        """Test errors raised by loading."""

        pc = PointCloud(self.fn_las, data_column="Z")
        pc.load()

        # Error if already loaded
        with pytest.raises(ValueError, match="Data are already loaded."):
            pc.load()

        pc = PointCloud(self.fn_las, data_column="Z")
        pc._name = None
        with pytest.raises(AttributeError, match="Cannot load as filename is not set anymore.*"):
            pc.load()

    def test_data_column(self) -> None:
        """Test the setting and getting of the main data column."""

        # Assert column is set properly at instantiation
        pc = PointCloud(self.gdf1, data_column="b1")
        assert pc.data_column == "b1"

        # And can be reset to another name if it exists
        pc2 = PointCloud(self.gdf2, data_column="b1")
        assert pc2.data_column == "b1"
        # First syntax
        pc2.data_column = "b2"
        assert pc2.data_column == "b2"
        # Equivalent syntax
        pc2.set_data_column("b1")
        assert pc2.data_column == "b1"

    def test_data_column__errors(self) -> None:
        """Test errors raised during setting of data column."""

        pc = PointCloud(self.gdf1, data_column="b1")
        # If the data column does not exist
        with pytest.raises(ValueError, match="Data column column_that_does_not_exist not found*"):
            pc.data_column = "column_that_does_not_exist"
        # Equivalent syntax
        with pytest.raises(ValueError, match="Data column column_that_does_not_exist not found*"):
            pc.set_data_column("column_that_does_not_exist")

    def test_pointcloud_equal(self) -> None:
        """Test pointcloud equality."""



    def test_from_array(self) -> None:
        """Test building point cloud from array."""

        # Build from array and compare
        pc1 = PointCloud(self.gdf1, data_column="b1")
        pc_from_arr = PointCloud.from_array(array=self.arr_points, crs=4326, data_column="b1")
        assert pc_from_arr.pointcloud_equal(pc1)

        # Should be the same witht transposed array
        pc_from_arr = PointCloud.from_array(array=self.arr_points.T, crs=4326, data_column="b1")
        assert pc_from_arr.pointcloud_equal(pc1)

    def test_from_array__errors(self):
        """Test errors raised during creation with array."""

        array = np.ones((4, 5))
        with pytest.raises(ValueError, match="Array must be of shape 3xN or Nx3."):
            PointCloud.from_array(array=array, crs=4326)

    def test_from_tuples(self) -> None:
        """Test building point cloud from list of tuples."""

        pc1 = PointCloud(self.gdf1, data_column="b1")
        tuples_xyz = list(zip(self.arr_points[:, 0], self.arr_points[:, 1], self.arr_points[:, 2]))
        pc_from_tuples = PointCloud.from_tuples(tuples_xyz=tuples_xyz, crs=4326, data_column="b1")
        assert pc_from_tuples.pointcloud_equal(pc1)

    def test_from_xyz(self) -> None:
        """Test building point cloud from xyz array-like."""

        # Build from array and compare
        pc1 = PointCloud(self.gdf1, data_column="b1")
        pc_from_xyz = PointCloud.from_xyz(x=self.arr_points[:, 0], y=self.arr_points[:, 1], z=self.arr_points[:, 2],
                                              crs=4326, data_column="b1")
        assert pc_from_xyz.pointcloud_equal(pc1)

        # Test with lists
        pc_from_xyz = PointCloud.from_xyz(x=list(self.arr_points[:, 0]),
                                          y=list(self.arr_points[:, 1]),
                                          z=list(self.arr_points[:, 2]),
                                        crs=4326,
                                        data_column="b1")
        assert pc_from_xyz.pointcloud_equal(pc1)

        # Test with tuples
        pc_from_xyz = PointCloud.from_xyz(x=tuple(self.arr_points[:, 0]),
                                            y=tuple(self.arr_points[:, 1]),
                                            z=tuple(self.arr_points[:, 2]),
                                            crs=4326,
                                            data_column="b1")
        assert pc_from_xyz.pointcloud_equal(pc1)

    def test_to_array(self) -> None:
        """Test exporting point cloud to array."""

        # Convert point cloud and compare
        pc1 = PointCloud(self.gdf1, data_column="b1")
        arr_from_pc = pc1.to_array()
        assert np.array_equal(arr_from_pc, self.arr_points.T)

    def test_to_tuples(self) -> None:
        """Test exporting point cloud to tuples."""

        # Convert point cloud and compare
        pc1 = PointCloud(self.gdf1, data_column="b1")
        tuples_xyz = list(zip(self.arr_points[:, 0], self.arr_points[:, 1], self.arr_points[:, 2]))
        tuples_from_pc = pc1.to_tuples()
        assert tuples_from_pc == tuples_xyz

    def test_to_xyz(self) -> None:
        """Test exporting point cloud to xyz arrays."""

        # Convert point cloud and compare
        pc1 = PointCloud(self.gdf1, data_column="b1")
        xyz_from_pc = pc1.to_xyz()
        assert np.array_equal(np.stack(xyz_from_pc), self.arr_points.T)