"""Test for PointCloud class."""

from __future__ import annotations

import os
import re
import tempfile
import warnings

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely import Polygon

import geoutils as gu
from geoutils import PointCloud
from geoutils._typing import NDArrayNum


class TestPointCloud:

    # 1/ Synthetic 2D points with main column and no auxiliary column
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(100, 3)) + rng.normal(0, 0.15, size=(100, 3))
    gdf1 = gpd.GeoDataFrame(
        data={"b1": arr_points[:, 2]}, geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1]), crs=4326
    )

    # 2/ Synthetic 2D points with main column and with auxiliary column
    arr_points2 = rng.integers(low=1, high=1000, size=(100, 4)) + rng.normal(0, 0.15, size=(100, 4))
    gdf2 = gpd.GeoDataFrame(
        data=arr_points2[:, 2:],
        columns=["b1", "b2"],
        geometry=gpd.points_from_xy(x=arr_points2[:, 0], y=arr_points2[:, 1]),
        crs=4326,
    )

    # 3/ Synthetic 3D points and with auxiliary column
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(100, 3)) + rng.normal(0, 0.15, size=(100, 3))
    gdf3 = gpd.GeoDataFrame(
        data=[],
        columns=[],
        geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1], z=arr_points[:, 2]),
        crs=4326,
    )
    # To test 3D with an extra column
    gdf32 = gpd.GeoDataFrame(
        data=arr_points2[:, 3:],
        columns=["b2"],
        geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1], z=arr_points[:, 2]),
        crs=4326,
    )

    # 4/ LAS file
    fn_las = gu.examples.get_path_test("coromandel_lidar")

    # 5/ Non-point vector (for error raising)
    poly = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    gdf4 = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

    def test_init(self) -> None:
        """Test instantiation of a point cloud."""

        # 1/ For a single column point cloud with 2D geometries
        pc = PointCloud(self.gdf1, data_column="b1")

        # Assert that both the dataframe and data column name are equal
        assert pc.data_column == "b1"
        assert_geodataframe_equal(pc.ds, self.gdf1)

        # 2/ For a point cloud with 3D geometries, no need to pass a data column
        pc = PointCloud(self.gdf3)

        # Assert that both the dataframe and data column name are equal
        assert pc.data_column is None
        assert_geodataframe_equal(pc.ds, self.gdf3)

    def test_init__las(self) -> None:
        # Import optional laspy or skip test
        pytest.importorskip("laspy")

        # 1/ For a point cloud from LAS/LAZ file, no need to pass a data column to get the default Z
        pc = PointCloud(self.fn_las)

        assert pc.data_column == "Z"
        assert not pc.is_loaded

        # 2/ But we can still specify another one
        pc = PointCloud(self.fn_las, data_column="number_of_returns")

        assert pc.data_column == "number_of_returns"
        assert not pc.is_loaded

    def test_init__errors(self) -> None:
        """Test errors raised by point cloud instantiation."""

        # If the data column does not exist
        with pytest.raises(ValueError, match="Data column column_that_does_not_exist not found*"):
            PointCloud(self.gdf1, data_column="column_that_does_not_exist")

        # If vector is not only comprised of points
        with pytest.raises(ValueError, match="This vector file contains non-point geometries*"):
            PointCloud(self.gdf4, data_column="z")

        # If the data column is not defined and the geometries are 2D
        with pytest.raises(ValueError, match="A data column name must be passed for a point cloud with 2D.*"):
            PointCloud(self.gdf1)

    def test_load__las(self) -> None:
        """
        Test loading of a point cloud (only possible with a LAS file).

        This test also serves to test the overridden methods "crs", "bounds", "nb_points", "nongeo_columns" in relation
        to loading.
        """

        # Import optional laspy or skip test
        pytest.importorskip("laspy")

        # 1/ Check unloaded and loaded attributes are all the same

        pc = PointCloud(self.fn_las)

        # Check point cloud is not loaded, and fetch before metadata
        assert not pc.is_loaded
        before_crs = pc.crs
        before_bounds = pc.bounds
        before_nb_points = pc.point_count
        before_columns = pc.columns

        # Load and fetch after metadata
        pc.load(columns="all")
        assert pc.is_loaded

        after_crs = pc.crs
        after_bounds = pc.bounds
        after_nb_points = pc.point_count
        after_columns = pc.columns

        # Check those are equal
        assert before_crs == after_crs
        assert before_bounds == after_bounds
        assert before_nb_points == after_nb_points
        assert before_columns == after_columns

        # 2/ Check default column argument
        pc = PointCloud(self.fn_las)
        pc.load()

        assert pc._nongeo_columns == ["Z"]

        # 3/ Check implicit loading when calling a function requiring .ds
        pc = PointCloud(self.fn_las)
        assert not pc.is_loaded

        pc.buffer(distance=0.1)
        assert pc.is_loaded

    def test_load__errors(self) -> None:
        """Test errors raised by loading."""

        # Import optional laspy or skip test
        pytest.importorskip("laspy")

        pc = PointCloud(self.fn_las)
        pc.load()

        # Error if already loaded
        with pytest.raises(ValueError, match="Data are already loaded."):
            pc.load()

        pc = PointCloud(self.fn_las)
        pc._name = None
        with pytest.raises(AttributeError, match="Cannot load as filename is not set anymore.*"):
            pc.load()

    def test_getitem_setitem(self) -> None:
        """Test the __getitem__ method ([]) for indexing and __setitem__ for index assignment."""

        # -- First, we test mask or boolean array indexing and assignment, specific to point clouds --

        # Open a point cloud
        # We need to do a deep copy to avoid modifying the original object
        pc = PointCloud(self.gdf1.copy(deep=True), data_column="b1")

        # Create a boolean array of the same shape, and a mask of the same transform/crs
        rng = np.random.default_rng(42)
        arr = rng.integers(low=0, high=2, size=pc.point_count, dtype=bool)
        mask = gu.PointCloud.from_xyz(x=pc.geometry.x.values, y=pc.geometry.y.values, z=arr, crs=pc.crs)

        # Check that indexing works with both of those
        vals_arr = pc[arr]
        vals_mask = pc[mask]

        # Those indexing operations should yield the same point cloud
        assert isinstance(vals_arr, PointCloud)
        assert isinstance(vals_mask, PointCloud)
        assert vals_mask.pointcloud_equal(vals_arr)

        # Now, we test index assignment
        pc2 = pc.copy()

        # It should work with a number, or a 1D array of the same length as the indexed one
        pc["b1"] = 1.0
        pc2["b1"] = np.ones(pc2.point_count)

        # The point clouds should be the same
        assert pc2.pointcloud_equal(pc)

        # -- Finally, we check that errors are raised for both indexing and index assignment --

        # For indexing
        op_name_index = "an indexing operation"
        message_pc = "Both point clouds must have the same points X/Y coordinates and CRS for {}."

        # An error when the georeferencing of the Mask does not match
        mask.translate(1, 1, inplace=True)
        with pytest.raises(ValueError, match=re.escape(message_pc.format(op_name_index))):
            pc[mask]

    def test_data_column(self) -> None:
        """Test the setting and getting of the main data column."""

        # Assert column is set properly at instantiation
        pc = PointCloud(self.gdf1, data_column="b1")
        assert pc.data_column == "b1"
        assert np.array_equal(pc.data, self.gdf1["b1"].values)

        # And can be reset to another name if it exists
        pc2 = PointCloud(self.gdf2, data_column="b1")
        assert pc2.data_column == "b1"
        assert np.array_equal(pc2.data, self.gdf2["b1"].values)

        # First syntax
        pc2.data_column = "b2"
        assert pc2.data_column == "b2"
        assert np.array_equal(pc2.data, self.gdf2["b2"].values)

        # Equivalent syntax
        pc2.set_data_column("b1")
        assert pc2.data_column == "b1"
        assert np.array_equal(pc2.data, self.gdf2["b1"].values)

        # Assert no data column is set for 3D points, using the Z coordinates instead
        pc3 = PointCloud(self.gdf3)
        assert pc3.data_column is None
        assert np.array_equal(pc3.data, self.gdf3.geometry.z.values)

    def test_data_column__errors(self) -> None:
        """Test errors raised during setting of data column."""

        pc = PointCloud(self.gdf1, data_column="b1")
        # If the data column does not exist
        with pytest.raises(ValueError, match="Data column column_that_does_not_exist not found*"):
            pc.data_column = "column_that_does_not_exist"
        # Equivalent syntax
        with pytest.raises(ValueError, match="Data column column_that_does_not_exist not found*"):
            pc.set_data_column("column_that_does_not_exist")

        # If a data column name is passed for 3D points
        with pytest.warns(UserWarning, match="Overriding 3D points with*"):
            pc4 = PointCloud(self.gdf32, data_column="b2")
            assert pc4.data_column == "b2"
            assert np.array_equal(pc4.data, self.gdf32["b2"].values)

    def test_data(self) -> None:
        """Test the setting and getting of the main data, depending on input geometry."""

        # For a point cloud using 2D geometries + a main data column
        pc = PointCloud(self.gdf1, data_column="b1")
        assert np.array_equal(pc.data, self.gdf1["b1"].values)
        pc += 1
        assert np.array_equal(pc.data, self.gdf1["b1"].values + 1)

        # For a point cloud using 3D geometries
        pc2 = PointCloud(self.gdf3)
        assert np.array_equal(pc2.data, self.gdf3.geometry.z.values)
        pc2 += 1
        assert np.array_equal(pc2.data, self.gdf3.geometry.z.values + 1)

    def test_from_array(self) -> None:
        """Test building point cloud from array."""

        # Build from array and compare
        pc1 = PointCloud(self.gdf1, data_column="b1")
        pc_from_arr = PointCloud.from_array(self.arr_points, crs=4326, data_column="b1")
        assert pc_from_arr.pointcloud_equal(pc1)

        # Should be the same with transposed array
        pc_from_arr = PointCloud.from_array(self.arr_points.T, crs=4326, data_column="b1")
        assert pc_from_arr.pointcloud_equal(pc1)

    def test_from_array__errors(self) -> None:
        """Test errors raised during creation with array."""

        array = np.ones((4, 5))
        with pytest.raises(ValueError, match="Array must be of shape 3xN or Nx3."):
            PointCloud.from_array(array, crs=4326)

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
        pc_from_xyz = PointCloud.from_xyz(
            x=self.arr_points[:, 0], y=self.arr_points[:, 1], z=self.arr_points[:, 2], crs=4326, data_column="b1"
        )
        assert pc_from_xyz.pointcloud_equal(pc1)

        # Test with lists
        pc_from_xyz = PointCloud.from_xyz(
            x=list(self.arr_points[:, 0]),
            y=list(self.arr_points[:, 1]),
            z=list(self.arr_points[:, 2]),
            crs=4326,
            data_column="b1",
        )
        assert pc_from_xyz.pointcloud_equal(pc1)

        # Test with tuples
        pc_from_xyz = PointCloud.from_xyz(
            x=tuple(self.arr_points[:, 0]),
            y=tuple(self.arr_points[:, 1]),
            z=tuple(self.arr_points[:, 2]),
            crs=4326,
            data_column="b1",
        )
        assert pc_from_xyz.pointcloud_equal(pc1)

        # Build with the use_z option to compare to 3D points
        pc3 = PointCloud(self.gdf3)
        pc3_from_xyz = PointCloud.from_xyz(
            x=self.arr_points[:, 0], y=self.arr_points[:, 1], z=self.arr_points[:, 2], crs=4326, use_z=True
        )
        assert pc3_from_xyz.pointcloud_equal(pc3)

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

    specific_method_args = {
        "reproject": {"crs": CRS.from_epsg(32610)},
        "crop": {
            "crop_geom": (
                np.min(gdf1.geometry.x.values),
                np.min(gdf1.geometry.y.values),
                np.max(gdf1.geometry.x.values),
                np.max(gdf1.geometry.y.values),
            )
        },
        "translate": {"xoff": 1, "yoff": 1},
        "set_precision": {"grid_size": 1},
        "to_crs": {"crs": CRS.from_epsg(32610)},
        "set_crs": {"crs": CRS.from_epsg(32610), "allow_override": True},
        "rename_geometry": {"col": "lol"},
    }

    def test_to_las(self) -> None:

        # Import optional laspy or skip test
        pytest.importorskip("laspy")

        # 1/ For a X/Y/Z point cloud with no auxiliary data
        pc1 = PointCloud(self.gdf1, data_column="b1")

        # Temporary folder
        temp_dir = tempfile.TemporaryDirectory()

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test.las")
        pc1.to_las(temp_file)
        saved1 = gu.PointCloud(temp_file)

        # Check loaded point cloud is equal to saved one
        atol = 0.1
        assert np.allclose(pc1.geometry.x.values, saved1.geometry.x.values, atol=atol)
        assert np.allclose(pc1.geometry.y.values, saved1.geometry.y.values, atol=atol)
        assert np.allclose(pc1.data, saved1.data, atol=atol)

        # 2/ For a X/Y/Z point cloud with some auxiliary data
        pc2 = PointCloud(self.gdf2, data_column="b1")

        # Save file to temporary file, with defaults opts
        temp_file = os.path.join(temp_dir.name, "test2.las")
        pc2.to_las(temp_file)
        saved2 = gu.PointCloud(temp_file)
        saved2.load(["Z", "b2"])

        # Check loaded point cloud is equal to saved one
        assert np.allclose(pc2.geometry.x.values, saved2.geometry.x.values, atol=atol)
        assert np.allclose(pc2.geometry.y.values, saved2.geometry.y.values, atol=atol)
        assert np.allclose(pc2.data, saved2.data, atol=atol)
        assert np.allclose(pc2["b2"].values, saved2["b2"].values, atol=atol)

    @pytest.mark.parametrize(
        "method", ["reproject", "crop", "translate", "set_precision", "to_crs", "set_crs", "rename_geometry"]
    )  # type: ignore
    def test_cast_vector_methods__geometry_invariant(self, method: str):
        """Test that method that don't modify geometry do cast back to a PointCloud."""

        pc1 = PointCloud(self.gdf1, data_column="b1")

        getattr(pc1, method)(**self.specific_method_args[method])
        assert isinstance(pc1, PointCloud)


class TestArithmetic:
    """
    Test that all arithmetic overloading functions work as expected.
    """

    # Create fake point clouds with random values in 0-255 and dtype uint8
    # TODO: Add the case where a mask exists in the array, as in test_data_setter
    rng = np.random.default_rng(42)
    nb_points = 20

    min_val = np.iinfo("uint8").min
    max_val = np.iinfo("uint8").max
    coords = rng.integers(min_val, max_val, (2, nb_points), dtype="uint8")
    data1 = rng.integers(min_val, max_val, nb_points, dtype="uint8")
    data2 = rng.integers(min_val, max_val, nb_points, dtype="uint8")

    # Create random masks
    mask1 = rng.integers(0, 2, size=nb_points, dtype=bool)
    mask2 = rng.integers(0, 2, size=nb_points, dtype=bool)
    mask3 = rng.integers(0, 2, size=nb_points, dtype=bool)

    crs = CRS(4326)
    wrong_crs = CRS(32610)

    # Create point clouds
    pc1 = gu.PointCloud.from_xyz(x=coords[0, :], y=coords[1, :], z=data1, crs=crs)
    pc2 = gu.PointCloud.from_xyz(x=coords[0, :], y=coords[1, :], z=data2, crs=crs)

    # Tests with different dtype
    pc1_f32 = gu.PointCloud.from_xyz(x=coords[0, :], y=coords[1, :], z=data1.astype("float32"), crs=crs)

    # With zero for operations such as division
    data_zeros = np.copy(data2)
    data_zeros[0:2] = 0
    pc2_zero = gu.PointCloud.from_xyz(x=coords[0, :], y=coords[1, :], z=data_zeros, crs=crs)

    pc1_wrong_crs = gu.PointCloud.from_xyz(x=coords[0, :], y=coords[1, :], z=data1, crs=wrong_crs)

    # Create rasters with different shape, crs or transforms for testing errors
    # Wrong shaped arrays to check errors are raised
    arr_wrong_shape = rng.integers(min_val, max_val, (3, nb_points - 1), dtype="int32") + rng.normal(
        size=(3, nb_points - 1)
    )
    mask_wrong_shape = rng.integers(0, 2, size=nb_points - 1, dtype=bool)

    # Wrong coordinate array to check errors are raised
    arr_wrong_coord = rng.integers(min_val, max_val, (3, nb_points), dtype="int32") + rng.normal(size=(3, nb_points))
    pc1_wrong_shape = gu.PointCloud.from_array(arr_wrong_shape, crs=crs)

    pc1_wrong_coords = gu.PointCloud.from_array(arr_wrong_coord, crs=crs)

    def test_pointcloud_equal(self) -> None:
        """
        Test that pointcloud_equal() works as expected.
        """
        pc1 = self.pc1
        pc2 = pc1.copy()
        assert pc1.pointcloud_equal(pc2)

        # Change data
        pc2.data = pc2.data + 1
        assert not pc1.pointcloud_equal(pc2)

        # Change dtype
        pc2 = pc1.copy()
        pc2 = pc2.astype("float32")
        assert not pc1.pointcloud_equal(pc2)

        # Change coordinates
        pc2 = pc1.copy()
        pc2 = pc2.translate(2, 1)
        assert not pc1.pointcloud_equal(pc2)

        # Change CRS
        pc2 = pc1.copy()
        pc2 = pc2.set_crs(self.wrong_crs, allow_override=True)
        assert not pc1.pointcloud_equal(pc2)

    def test_georeferenced_coords_equal(self) -> None:
        """
        Test that equal for shape, crs and transform work as expected
        """

        # -- Test 1: based on a copy --
        pc1 = self.pc1
        pc2 = pc1.copy()
        assert pc1.georeferenced_coords_equal(pc2)

        # Change data
        pc2.data = pc2.data + 1
        assert pc1.georeferenced_coords_equal(pc2)

        # Change dtype
        pc2 = pc1.copy()
        pc2 = pc2.astype("float32")
        assert pc1.georeferenced_coords_equal(pc2)

        # Change coords
        pc2 = pc1.copy()
        pc2 = pc2.translate(2, 1)
        assert not pc1.georeferenced_coords_equal(pc2)

        # Change CRS
        pc2 = pc1.copy()
        pc2 = pc2.set_crs(self.wrong_crs, allow_override=True)
        assert not pc1.georeferenced_coords_equal(pc2)

        # -- Test 2: based on another Raster with one different georeferenced grid attribute --

        assert not pc1.georeferenced_coords_equal(self.pc1_wrong_crs)

        assert not pc1.georeferenced_coords_equal(self.pc1_wrong_shape)

        assert not pc1.georeferenced_coords_equal(self.pc1_wrong_coords)

    # List of operations with two operands
    ops_2args = [
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__floordiv__",
        "__rfloordiv__",
        "__mod__",
    ]

    @pytest.mark.parametrize("op", ops_2args)  # type: ignore
    def test_ops_2args_expl(self, op: str) -> None:
        """
        Check that arithmetic overloading functions, with two operands, work as expected when called explicitly.
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Point clouds with different dtypes, np.ndarray, single number
        pc1 = self.pc1
        pc1_f32 = self.pc1_f32
        pc2 = self.pc2
        pc2_zero = self.pc2_zero
        rng = np.random.default_rng(42)
        array = rng.integers(1, 255, self.nb_points).astype("float64")
        floatval = 3.14
        intval = 1

        # Test with 2 uint8 point clouds
        pc1 = self.pc1
        pc2 = self.pc2
        pc3 = getattr(pc1, op)(pc2)
        ctype = np.promote_types(pc1.data.dtype, pc2.data.dtype)
        numpy_output = getattr(pc1.data.astype(ctype), op)(pc2.data.astype(ctype))
        assert isinstance(pc3, gu.PointCloud)
        assert np.all(pc3.data == numpy_output)
        assert pc3.data.dtype == numpy_output.dtype
        assert pc3.crs == pc1.crs

        # Test original data are not modified
        pc1_copy = pc1.copy()
        pc2_copy = pc2.copy()
        pc3 = getattr(pc1, op)(pc2)
        assert isinstance(pc3, gu.PointCloud)
        assert pc1.pointcloud_equal(pc1_copy)
        assert pc2.pointcloud_equal(pc2_copy)

        # Test with different dtypes
        pc1 = self.pc1_f32
        pc2 = self.pc2
        pc3 = getattr(pc1_f32, op)(pc2)
        assert pc3.data.dtype == np.dtype("float32")
        assert np.all(pc3.data == getattr(pc1.data, op)(pc2.data))

        # Test with zeros values (e.g. division)
        pc1 = self.pc1
        pc3 = getattr(pc1, op)(pc2_zero)
        assert np.all(pc3.data == getattr(pc1.data, op)(pc2_zero.data))

        # Test with a numpy array
        pc1 = self.pc1_f32
        pc3 = getattr(pc1, op)(array)
        assert isinstance(pc3, gu.PointCloud)
        assert np.all(pc3.data == getattr(pc1.data, op)(array))

        # Test with an integer
        pc3 = getattr(pc1, op)(intval)
        assert isinstance(pc3, gu.PointCloud)
        assert np.all(pc3.data == getattr(pc1.data, op)(intval))

        # Test with a float value
        pc3 = getattr(pc1, op)(floatval)
        assert isinstance(pc3, gu.PointCloud)
        # Behaviour is more complex for scalars since NumPy 2.0,
        # so simply comparing it is consistent with that of masked arrays
        assert pc3.pointcloud_equal(self.copy(getattr(pc1.data, op)(floatval), pc_ref=pc3))

    reflective_ops = [["__add__", "__radd__"], ["__mul__", "__rmul__"]]

    @pytest.mark.parametrize("ops", reflective_ops)  # type: ignore
    def test_reflectivity(self, ops: list[str]) -> None:
        """
        Check reflective operations
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray, single number
        rng = np.random.default_rng(42)
        array = rng.integers(1, 255, self.nb_points).astype("float64")
        floatval = 3.14
        intval = 1

        # Get reflective operations
        op1, op2 = ops

        # Test with uint8 rasters
        pc3 = getattr(self.pc1, op1)(self.pc2)
        pc4 = getattr(self.pc1, op2)(self.pc2)
        assert pc3.pointcloud_equal(pc4)

        # Test with different dtypes
        pc3 = getattr(self.pc1_f32, op1)(self.pc2)
        pc4 = getattr(self.pc1_f32, op2)(self.pc2)
        assert pc3.pointcloud_equal(pc4)

        # Test with zeros values (e.g. division)
        pc3 = getattr(self.pc1, op1)(self.pc2_zero)
        pc4 = getattr(self.pc1, op2)(self.pc2_zero)
        assert pc3.pointcloud_equal(pc4)

        # Test with a numpy array
        pc3 = getattr(self.pc1, op1)(array)
        pc4 = getattr(self.pc1, op2)(array)
        assert pc3.pointcloud_equal(pc4)

        # Test with an integer
        pc3 = getattr(self.pc1, op1)(intval)
        pc4 = getattr(self.pc1, op2)(intval)
        assert pc3.pointcloud_equal(pc4)

        # Test with a float value
        pc3 = getattr(self.pc1, op1)(floatval)
        pc4 = getattr(self.pc1, op2)(floatval)
        assert pc3.pointcloud_equal(pc4)

    @classmethod
    def copy(
        cls: type[TestArithmetic],
        data: NDArrayNum,
        pc_ref: gu.PointCloud,
    ) -> gu.PointCloud:
        """
        Generate a pointcloud from numpy array, with set georeferencing. Used for testing only.
        """

        return pc_ref.copy(new_array=data)

    def test_ops_2args_implicit(self) -> None:
        """
        Test certain arithmetic overloading when called with symbols (+, -, *, /, //, %).
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray with 2D or 3D shape, single number
        pc1 = self.pc1
        pc1_f32 = self.pc1_f32
        pc2 = self.pc2
        rng = np.random.default_rng(42)
        array_2d = rng.integers(1, 255, (1, self.nb_points)).astype("uint8")
        array_1d = rng.integers(1, 255, self.nb_points).astype("uint8")
        floatval = 3.14

        # Addition
        assert (pc1 + pc2).pointcloud_equal(self.copy(pc1.data + pc2.data, pc_ref=pc1))
        assert (pc1_f32 + pc2).pointcloud_equal(self.copy(pc1_f32.data + pc2.data, pc_ref=pc1))
        assert (array_2d + pc2).pointcloud_equal(self.copy(array_2d + pc2.data, pc_ref=pc2))
        assert (pc2 + array_2d).pointcloud_equal(self.copy(pc2.data + array_2d, pc_ref=pc2))
        assert (array_1d + pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] + pc2.data, pc_ref=pc2))
        assert (pc2 + array_1d).pointcloud_equal(self.copy(pc2.data + array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 + floatval).pointcloud_equal(self.copy(pc1.data + floatval, pc_ref=pc1))
        assert (floatval + pc1).pointcloud_equal(self.copy(floatval + pc1.data, pc_ref=pc1))
        assert (pc1 + pc2).pointcloud_equal(pc2 + pc1)

        # Multiplication
        assert (pc1 * pc2).pointcloud_equal(self.copy(pc1.data * pc2.data, pc_ref=pc1))
        assert (pc1_f32 * pc2).pointcloud_equal(self.copy(pc1_f32.data * pc2.data, pc_ref=pc1))
        assert (array_2d * pc2).pointcloud_equal(self.copy(array_2d * pc2.data, pc_ref=pc2))
        assert (pc2 * array_2d).pointcloud_equal(self.copy(pc2.data * array_2d, pc_ref=pc2))
        assert (array_1d * pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] * pc2.data, pc_ref=pc2))
        assert (pc2 * array_1d).pointcloud_equal(self.copy(pc2.data * array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 * floatval).pointcloud_equal(self.copy(pc1.data * floatval, pc_ref=pc1))
        assert (floatval * pc1).pointcloud_equal(self.copy(floatval * pc1.data, pc_ref=pc1))
        assert (pc1 * pc2).pointcloud_equal(pc2 * pc1)

        # Subtraction
        assert (pc1 - pc2).pointcloud_equal(self.copy(pc1.data - pc2.data, pc_ref=pc1))
        assert (pc1_f32 - pc2).pointcloud_equal(self.copy(pc1_f32.data - pc2.data, pc_ref=pc1))
        assert (array_2d - pc2).pointcloud_equal(self.copy(array_2d - pc2.data, pc_ref=pc2))
        assert (pc2 - array_2d).pointcloud_equal(self.copy(pc2.data - array_2d, pc_ref=pc2))
        assert (array_1d - pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] - pc2.data, pc_ref=pc2))
        assert (pc2 - array_1d).pointcloud_equal(self.copy(pc2.data - array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 - floatval).pointcloud_equal(self.copy(pc1.data - floatval, pc_ref=pc1))
        assert (floatval - pc1).pointcloud_equal(self.copy(floatval - pc1.data, pc_ref=pc1))

        # True division
        assert (pc1 / pc2).pointcloud_equal(self.copy(pc1.data / pc2.data, pc_ref=pc1))
        assert (pc1_f32 / pc2).pointcloud_equal(self.copy(pc1_f32.data / pc2.data, pc_ref=pc1))
        assert (array_2d / pc2).pointcloud_equal(self.copy(array_2d / pc2.data, pc_ref=pc2))
        assert (pc2 / array_2d).pointcloud_equal(self.copy(pc2.data / array_2d, pc_ref=pc2))
        assert (array_1d / pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] / pc2.data, pc_ref=pc1))
        assert (pc2 / array_1d).pointcloud_equal(self.copy(pc2.data / array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 / floatval).pointcloud_equal(self.copy(pc1.data / floatval, pc_ref=pc1))
        assert (floatval / pc1).pointcloud_equal(self.copy(floatval / pc1.data, pc_ref=pc1))

        # Floor division
        assert (pc1 // pc2).pointcloud_equal(self.copy(pc1.data // pc2.data, pc_ref=pc1))
        assert (pc1_f32 // pc2).pointcloud_equal(self.copy(pc1_f32.data // pc2.data, pc_ref=pc1))
        assert (array_2d // pc2).pointcloud_equal(self.copy(array_2d // pc2.data, pc_ref=pc1))
        assert (pc2 // array_2d).pointcloud_equal(self.copy(pc2.data // array_2d, pc_ref=pc1))
        assert (array_1d // pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] // pc2.data, pc_ref=pc1))
        assert (pc2 // array_1d).pointcloud_equal(self.copy(pc2.data // array_1d[np.newaxis, :], pc_ref=pc1))
        assert (pc1 // floatval).pointcloud_equal(self.copy(pc1.data // floatval, pc_ref=pc1))
        assert (floatval // pc1).pointcloud_equal(self.copy(floatval // pc1.data, pc_ref=pc1))

        # Modulo
        assert (pc1 % pc2).pointcloud_equal(self.copy(pc1.data % pc2.data, pc_ref=pc1))
        assert (pc1_f32 % pc2).pointcloud_equal(self.copy(pc1_f32.data % pc2.data, pc_ref=pc1))
        assert (array_2d % pc2).pointcloud_equal(self.copy(array_2d % pc2.data, pc_ref=pc1))
        assert (pc2 % array_2d).pointcloud_equal(self.copy(pc2.data % array_2d, pc_ref=pc1))
        assert (array_1d % pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] % pc2.data, pc_ref=pc1))
        assert (pc2 % array_1d).pointcloud_equal(self.copy(pc2.data % array_1d[np.newaxis, :], pc_ref=pc1))
        assert (pc1 % floatval).pointcloud_equal(self.copy(pc1.data % floatval, pc_ref=pc1))

    def test_ops_logical_implicit(self) -> None:
        """
        Test logical arithmetic overloading when called with symbols (==, !=, <, <=, >, >=).
        """
        warnings.filterwarnings("ignore", message="invalid value encountered")

        # Test various inputs: Raster with different dtypes, np.ndarray with 2D or 3D shape, single number
        pc1 = self.pc1
        pc1_f32 = self.pc1_f32
        pc2 = self.pc2
        rng = np.random.default_rng(42)
        array_2d = rng.integers(1, 255, (1, self.nb_points)).astype("uint8")
        array_1d = rng.integers(1, 255, self.nb_points).astype("uint8")
        floatval = 3.14

        # Equality
        assert (pc1 == pc2).pointcloud_equal(self.copy(pc1.data == pc2.data, pc_ref=pc1))
        assert (pc1_f32 == pc2).pointcloud_equal(self.copy(pc1_f32.data == pc2.data, pc_ref=pc1))
        assert (array_2d == pc2).pointcloud_equal(self.copy(array_2d == pc2.data, pc_ref=pc2))
        assert (pc2 == array_2d).pointcloud_equal(self.copy(pc2.data == array_2d, pc_ref=pc2))
        assert (array_1d == pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] == pc2.data, pc_ref=pc2))
        assert (pc2 == array_1d).pointcloud_equal(self.copy(pc2.data == array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 == floatval).pointcloud_equal(self.copy(pc1.data == floatval, pc_ref=pc1))
        assert (floatval == pc1).pointcloud_equal(self.copy(floatval == pc1.data, pc_ref=pc1))
        assert (pc1 == pc2).pointcloud_equal(pc2 == pc1)

        # Non-equality
        assert (pc1 != pc2).pointcloud_equal(self.copy(pc1.data != pc2.data, pc_ref=pc1))
        assert (pc1_f32 != pc2).pointcloud_equal(self.copy(pc1_f32.data != pc2.data, pc_ref=pc1))
        assert (array_2d != pc2).pointcloud_equal(self.copy(array_2d != pc2.data, pc_ref=pc2))
        assert (pc2 != array_2d).pointcloud_equal(self.copy(pc2.data != array_2d, pc_ref=pc2))
        assert (array_1d != pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] != pc2.data, pc_ref=pc2))
        assert (pc2 != array_1d).pointcloud_equal(self.copy(pc2.data != array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 != floatval).pointcloud_equal(self.copy(pc1.data != floatval, pc_ref=pc1))
        assert (floatval != pc1).pointcloud_equal(self.copy(floatval != pc1.data, pc_ref=pc1))
        assert (pc1 != pc2).pointcloud_equal(pc2 != pc1)

        # Lower than
        assert (pc1 < pc2).pointcloud_equal(self.copy(pc1.data < pc2.data, pc_ref=pc1))
        assert (pc1_f32 < pc2).pointcloud_equal(self.copy(pc1_f32.data < pc2.data, pc_ref=pc1))
        assert (array_2d < pc2).pointcloud_equal(self.copy(array_2d < pc2.data, pc_ref=pc2))
        assert (pc2 < array_2d).pointcloud_equal(self.copy(pc2.data < array_2d, pc_ref=pc2))
        assert (array_1d < pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] < pc2.data, pc_ref=pc2))
        assert (pc2 < array_1d).pointcloud_equal(self.copy(pc2.data < array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 < floatval).pointcloud_equal(self.copy(pc1.data < floatval, pc_ref=pc1))
        assert (floatval < pc1).pointcloud_equal(self.copy(floatval < pc1.data, pc_ref=pc1))

        # Lower equal
        assert (pc1 <= pc2).pointcloud_equal(self.copy(pc1.data <= pc2.data, pc_ref=pc1))
        assert (pc1_f32 <= pc2).pointcloud_equal(self.copy(pc1_f32.data <= pc2.data, pc_ref=pc1))
        assert (array_2d <= pc2).pointcloud_equal(self.copy(array_2d <= pc2.data, pc_ref=pc2))
        assert (pc2 <= array_2d).pointcloud_equal(self.copy(pc2.data <= array_2d, pc_ref=pc2))
        assert (array_1d <= pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] <= pc2.data, pc_ref=pc1))
        assert (pc2 <= array_1d).pointcloud_equal(self.copy(pc2.data <= array_1d[np.newaxis, :], pc_ref=pc2))
        assert (pc1 <= floatval).pointcloud_equal(self.copy(pc1.data <= floatval, pc_ref=pc1))
        assert (floatval <= pc1).pointcloud_equal(self.copy(floatval <= pc1.data, pc_ref=pc1))

        # Greater than
        assert (pc1 > pc2).pointcloud_equal(self.copy(pc1.data > pc2.data, pc_ref=pc1))
        assert (pc1_f32 > pc2).pointcloud_equal(self.copy(pc1_f32.data > pc2.data, pc_ref=pc1))
        assert (array_2d > pc2).pointcloud_equal(self.copy(array_2d > pc2.data, pc_ref=pc1))
        assert (pc2 > array_2d).pointcloud_equal(self.copy(pc2.data > array_2d, pc_ref=pc1))
        assert (array_1d > pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] > pc2.data, pc_ref=pc1))
        assert (pc2 > array_1d).pointcloud_equal(self.copy(pc2.data > array_1d[np.newaxis, :], pc_ref=pc1))
        assert (pc1 > floatval).pointcloud_equal(self.copy(pc1.data > floatval, pc_ref=pc1))
        assert (floatval > pc1).pointcloud_equal(self.copy(floatval > pc1.data, pc_ref=pc1))

        # Greater equal
        assert (pc1 >= pc2).pointcloud_equal(self.copy(pc1.data >= pc2.data, pc_ref=pc1))
        assert (pc1_f32 >= pc2).pointcloud_equal(self.copy(pc1_f32.data >= pc2.data, pc_ref=pc1))
        assert (array_2d >= pc2).pointcloud_equal(self.copy(array_2d >= pc2.data, pc_ref=pc1))
        assert (pc2 >= array_2d).pointcloud_equal(self.copy(pc2.data >= array_2d, pc_ref=pc1))
        assert (array_1d >= pc2).pointcloud_equal(self.copy(array_1d[np.newaxis, :] >= pc2.data, pc_ref=pc1))
        assert (pc2 >= array_1d).pointcloud_equal(self.copy(pc2.data >= array_1d[np.newaxis, :], pc_ref=pc1))
        assert (pc1 >= floatval).pointcloud_equal(self.copy(pc1.data >= floatval, pc_ref=pc1))

    def test_ops_logical_bitwise_implicit(self) -> None:
        # Create two masks
        pc1 = self.pc1
        m1 = self.pc1 > 128
        m2 = self.pc2 > 128
        rng = np.random.default_rng(42)
        array_1d = rng.integers(1, 255, self.nb_points).astype("uint8") > 128

        # Bitwise or
        assert (m1 | m2).pointcloud_equal(self.copy(m1.data | m2.data, pc_ref=pc1))
        assert (m1 | array_1d).pointcloud_equal(self.copy(m1.data | array_1d, pc_ref=pc1))
        assert (array_1d | m1).pointcloud_equal(self.copy(array_1d | m1.data, pc_ref=pc1))

        # Bitwise and
        assert (m1 & m2).pointcloud_equal(self.copy(m1.data & m2.data, pc_ref=pc1))
        assert (m1 & array_1d).pointcloud_equal(self.copy(m1.data & array_1d, pc_ref=pc1))
        assert (array_1d & m1).pointcloud_equal(self.copy(array_1d & m1.data, pc_ref=pc1))

        # Bitwise xor
        assert (m1 ^ m2).pointcloud_equal(self.copy(m1.data ^ m2.data, pc_ref=pc1))
        assert (m1 ^ array_1d).pointcloud_equal(self.copy(m1.data ^ array_1d, pc_ref=pc1))
        assert (array_1d ^ m1).pointcloud_equal(self.copy(array_1d ^ m1.data, pc_ref=pc1))

        # Bitwise invert
        assert (~m1).pointcloud_equal(self.copy(~m1.data, pc_ref=pc1))

    @pytest.mark.parametrize("op", ops_2args)  # type: ignore
    def test_raise_errors(self, op: str) -> None:
        """
        Test that errors are properly raised in certain situations.

        !! Important !! Here we test errors with the operator on the raster only (arithmetic overloading),
        calling with array first is supported with the NumPy interface and tested in ArrayInterface.
        """
        # Rasters with different CRS, transform, or shape
        # Different shape
        expected_message = (
            "Both point clouds must have the same points X/Y coordinates and CRS for an arithmetic operation."
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.pc2, op)(self.pc1_wrong_shape)

        # Different CRS
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.pc2, op)(self.pc1_wrong_crs)

        # Different transform
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.pc2, op)(self.pc1_wrong_coords)

        # Array with different shape
        expected_message = (
            "The array must be 1-dimensional with the same number of points as the point cloud for an arithmetic "
            "operation."
        )

        # Different shape, normal array with NaNs
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            getattr(self.pc2, op)(self.pc1_wrong_shape.data)

        # Wrong type of "other"
        expected_message = "Operation between an object of type .* and a point cloud impossible."
        with pytest.raises(NotImplementedError, match=expected_message):
            getattr(self.pc1, op)("some_string")

    @pytest.mark.parametrize("power", [2, 3.14, -1])  # type: ignore
    def test_power(self, power: float | int) -> None:
        if power > 0:  # Integers to negative integer powers are not allowed.
            assert self.pc1**power == self.copy(self.pc1.data**power, pc_ref=self.pc1)
        assert self.pc1_f32**power == self.copy(self.pc1_f32.data**power, pc_ref=self.pc1_f32)

    @pytest.mark.parametrize("dtype", ["float32", "uint8", "int32"])  # type: ignore
    def test_numpy_functions(self, dtype: str) -> None:
        """Test how rasters can be used as/with numpy arrays."""
        warnings.simplefilter("error")

        # Create an array of unique values starting at 0 and ending at 24
        array = np.arange(25, dtype=dtype)
        # Create an associated dummy coordinates
        coords = np.zeros((2, 25))

        # Create a raster from the array
        pc = gu.PointCloud.from_xyz(x=coords[0, :], y=coords[1, :], z=array, crs=4326)

        # Test some ufuncs
        assert np.median(pc) == 12.0
        assert np.mean(pc) == 12.0

        # Check that rasters don't become arrays when using simple arithmetic.
        assert isinstance(pc + 1, gu.PointCloud)

        # Test the data setter method by creating a new array
        pc.data = array + 2

        # Check that the median updated accordingly.
        assert np.median(pc) == 14.0

        # Test
        pc += array

        assert isinstance(pc, gu.PointCloud)
        assert np.median(pc) == 26.0


class TestArrayInterface:
    """Test that the array interface of PointCloud works as expected for ufuncs and array functions"""

    # -- First, we list all universal NumPy functions, or "ufuncs" --

    # All universal functions of NumPy, about 90 in 2022. See list: https://numpy.org/doc/stable/reference/ufuncs.html
    ufuncs_str = [
        ufunc
        for ufunc in np._core.umath.__all__
        if (
            ufunc[0] != "_"
            and ufunc.islower()
            and "err" not in ufunc
            and ufunc not in ["e", "pi", "frompyfunc", "euler_gamma", "vecdot", "vecmat", "matvec"]
        )
    ]

    # Universal functions with one input argument and one output, corresponding to (in NumPy 1.22.4):
    # ['absolute', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cbrt', 'ceil', 'conj', 'conjugate',
    # 'cos', 'cosh', 'deg2rad', 'degrees', 'exp', 'exp2', 'expm1', 'fabs', 'floor', 'invert', 'isfinite', 'isinf',
    # 'isnan', 'isnat', 'log', 'log10', 'log1p', 'log2', 'logical_not', 'negative', 'positive', 'rad2deg', 'radians',
    # 'reciprocal', 'rint', 'sign', 'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square', 'tan', 'tanh', 'trunc']
    ufuncs_str_1nin_1nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 1 and getattr(np, ufunc).nout == 1)
    ]

    # Universal functions with one input argument and two output (Note: none exist for three outputs or above)
    # Those correspond to: ['frexp', 'modf']
    ufuncs_str_1nin_2nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 1 and getattr(np, ufunc).nout == 2)
    ]

    # Universal functions with two input arguments and one output, corresponding to:
    # ['add', 'arctan2', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'copysign', 'divide', 'equal', 'floor_divide',
    #  'float_power', 'fmax', 'fmin', 'fmod', 'gcd', 'greater', 'greater_equal', 'heaviside', 'hypot', 'lcm', 'ldexp',
    #  'left_shift', 'less', 'less_equal', 'logaddexp', 'logaddexp2', 'logical_and', 'logical_or', 'logical_xor',
    #  'maximum', 'minimum', 'mod', 'multiply', 'nextafter', 'not_equal', 'power', 'remainder', 'right_shift',
    #  'subtract', 'true_divide']
    ufuncs_str_2nin_1nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 2 and getattr(np, ufunc).nout == 1)
    ]

    # Universal functions with two input arguments and two outputs (Note: none exist for three outputs or above)
    # These correspond to: ['divmod']
    ufuncs_str_2nin_2nout = [
        ufunc for ufunc in ufuncs_str if (getattr(np, ufunc).nin == 2 and getattr(np, ufunc).nout == 2)
    ]

    # -- Second, we list array functions we intend to support in the array interface --

    # To my knowledge, there is no list that includes all numpy functions (and we probably don't want to test them all)
    # Let's include manually the important ones:
    # - statistics: normal and for NaNs;
    # - sorting and counting;
    # Most other math functions are already universal functions

    # Separate between two lists (single input and double input) for testing
    handled_functions_2in = gu.pointcloud.pointcloud._HANDLED_FUNCTIONS_2NIN
    handled_functions_1in = gu.pointcloud.pointcloud._HANDLED_FUNCTIONS_1NIN

    # Details below:
    # NaN functions: [f for f in np.lib.nanfunctions.__all__]
    # nanstatfuncs = ['nansum', 'nanmax', 'nanmin', 'nanargmax', 'nanargmin', 'nanmean', 'nanmedian', 'nanpercentile',
    #             'nanvar', 'nanstd', 'nanprod', 'nancumsum', 'nancumprod', 'nanquantile']

    # Statistics and sorting matching NaN functions: https://numpy.org/doc/stable/reference/routines.statistics.html
    # and https://numpy.org/doc/stable/reference/routines.sort.html
    # statfuncs = ['sum', 'max', 'min', 'argmax', 'argmin', 'mean', 'median', 'percentile', 'var', 'std', 'prod',
    #              'cumsum', 'cumprod', 'quantile']

    # Sorting and counting ounting with single array input:
    # sortfuncs = ['sort', 'count_nonzero', 'unique]

    # --  Third, we define the test data --

    # We create two random array of varying dtype
    nb_points = 20
    min_val = np.iinfo("int32").min
    max_val = np.iinfo("int32").max
    rng = np.random.default_rng(42)
    # All points cloud should have the same X/Y for operations to work, we only change the data
    coords = rng.integers(min_val, max_val, (2, nb_points), dtype="int32") + rng.normal(size=(2, nb_points))
    data1 = rng.integers(min_val, max_val, nb_points, dtype="int32") + rng.normal(size=nb_points)
    data2 = rng.integers(min_val, max_val, nb_points, dtype="int32") + rng.normal(size=nb_points)
    # This third one is to try ufunc methods like reduce()
    data3 = rng.integers(min_val, max_val, nb_points, dtype="int32") + rng.normal(size=nb_points)

    # Create random masks
    mask1 = rng.integers(0, 2, size=nb_points, dtype=bool)
    mask2 = rng.integers(0, 2, size=nb_points, dtype=bool)
    mask3 = rng.integers(0, 2, size=nb_points, dtype=bool)

    # Assert that there is at least one unmasked value
    assert np.count_nonzero(~mask1) > 0
    assert np.count_nonzero(~mask2) > 0
    assert np.count_nonzero(~mask3) > 0

    # CRS
    crs = CRS(4326)

    # Wrong shaped arrays to check errors are raised
    arr_wrong_shape = rng.integers(min_val, max_val, (3, nb_points - 1), dtype="int32") + rng.normal(
        size=(3, nb_points - 1)
    )
    mask_wrong_shape = rng.integers(0, 2, size=nb_points - 1, dtype=bool)

    # Wrong coordinate array to check errors are raised
    arr_wrong_coord = rng.integers(min_val, max_val, (3, nb_points), dtype="int32") + rng.normal(size=(3, nb_points))

    # Wrong CRS input
    wrong_crs = CRS(32610)

    @pytest.mark.parametrize("ufunc_str", ufuncs_str_1nin_1nout + ufuncs_str_1nin_2nout)  # type: ignore
    @pytest.mark.parametrize("dtype", ["uint8", "int8", "float32"])  # type: ignore
    def test_array_ufunc_1nin_1nout(self, ufunc_str: str, dtype: str) -> None:
        """Test that ufuncs with one input and one output consistently return the same result as for masked arrays."""

        data = self.data1.astype(dtype)
        # If floating type, create NaNs
        if np.issubdtype(dtype, np.floating):
            data[self.mask1] = np.nan
        pc = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data, crs=self.crs)

        # Get ufunc
        ufunc = getattr(np, ufunc_str)

        # Find the common dtype between the point cloud and the most constrained input type (first character is input)
        try:
            com_dtype = np.promote_types(dtype, ufunc.types[0][0])
        # The promote_types function raises an error for object dtypes (previously returned by find_common_dtypes)
        # (TypeError needed for backwards compatibility; also exceptions.DTypePromotionError for NumPy 1.25 and above)
        except TypeError:
            com_dtype = np.dtype("O")

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            print(pc.data)

            # Check if our input dtype is possible on this ufunc, if yes check that outputs are identical
            if com_dtype in [str(np.dtype(t[0])) for t in ufunc.types]:  # noqa
                # For a single output
                if ufunc.nout == 1:
                    assert np.array_equal(ufunc(pc.data), ufunc(pc).data, equal_nan=True)

                # For two outputs
                elif ufunc.nout == 2:
                    outputs_pc = ufunc(pc)
                    output_arr = ufunc(pc.data)
                    assert np.array_equal(output_arr[0], outputs_pc[0].data, equal_nan=True) and np.array_equal(
                        output_arr[1], outputs_pc[1].data, equal_nan=True
                    )

            # If the input dtype is not possible, check that NumPy raises a TypeError
            else:
                with pytest.raises(TypeError):
                    ufunc(pc.data)
                with pytest.raises(TypeError):
                    ufunc(pc)

    @pytest.mark.parametrize("ufunc_str", ufuncs_str_2nin_1nout + ufuncs_str_2nin_2nout)  # type: ignore
    @pytest.mark.parametrize("dtype1", ["uint8", "int8", "float32"])  # type: ignore
    @pytest.mark.parametrize("dtype2", ["uint8", "int8", "float32"])  # type: ignore
    def test_array_ufunc_2nin_1nout(self, ufunc_str: str, dtype1: str, dtype2: str) -> None:
        """Test that ufuncs with two input arguments consistently return the same result as for masked arrays."""

        data1 = self.data1.astype(dtype1)
        data2 = self.data2.astype(dtype2)
        # If floating type, create NaNs
        if np.issubdtype(dtype1, np.floating):
            data1[self.mask1] = np.nan
        if np.issubdtype(dtype2, np.floating):
            data2[self.mask2] = np.nan

        pc1 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data1, crs=self.crs)
        pc2 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data2, crs=self.crs)

        ufunc = getattr(np, ufunc_str)

        # Find common dtype between the point cloud and the most constrained input type (first character is the input)
        try:
            com_dtype1 = np.promote_types(dtype1, ufunc.types[0][0])
        # The promote_types function raises an error for object dtypes (previously returned by find_common_dtypes)
        # (TypeError needed for backwards compatibility; also exceptions.DTypePromotionError for NumPy 1.25 and above)
        except TypeError:
            com_dtype1 = np.dtype("O")

        try:
            com_dtype2 = np.promote_types(dtype2, ufunc.types[0][1])
        # The promote_types function raises an error for object dtypes (previously returned by find_common_dtypes)
        # (TypeError needed for backwards compatibility; also exceptions.DTypePromotionError for NumPy 1.25 and above)
        except TypeError:
            com_dtype2 = np.dtype("O")

        # If the two input types can be the same type, pass a tuple with the common type of both
        # Below we ignore datetime and timedelta types "m" and "M", and int64 types "q" and "Q"
        if all(t[0] == t[1] for t in ufunc.types if not any(x in t[0:2] for x in ["m", "M", "q", "Q"])):
            try:
                com_dtype_both = np.promote_types(com_dtype1, com_dtype2)
            except TypeError:
                com_dtype_both = np.dtype("O")
            com_dtype_tuple = (com_dtype_both, com_dtype_both)

        # Otherwise, pass the tuple with each common type
        else:
            com_dtype_tuple = (com_dtype1, com_dtype2)

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # Check if both our input dtypes are possible on this ufunc, if yes check that outputs are identical
            if com_dtype_tuple in [(np.dtype(t[0]), np.dtype(t[1])) for t in ufunc.types]:  # noqa
                # For a single output
                if ufunc.nout == 1:
                    # There exists a single exception due to negative integers as exponent of integers in "power"
                    if ufunc_str == "power" and "int" in dtype1 and "int" in dtype2 and np.min(pc2.data) < 0:
                        with pytest.raises(ValueError, match="Integers to negative integer powers are not allowed."):
                            ufunc(pc1, pc2)
                        with pytest.raises(ValueError, match="Integers to negative integer powers are not allowed."):
                            ufunc(pc1.data, pc2.data)

                    # Otherwise, run the normal assertion for a single output
                    else:
                        assert np.array_equal(ufunc(pc1.data, pc2.data), ufunc(pc1, pc2).data, equal_nan=True)

                # For two outputs
                elif ufunc.nout == 2:
                    outputs_pc = ufunc(pc1, pc2)
                    output_arr = ufunc(pc1.data, pc2.data)
                    assert np.array_equal(output_arr[0], outputs_pc[0].data, equal_nan=True) and np.array_equal(
                        output_arr[1], outputs_pc[1].data, equal_nan=True
                    )

            # If the input dtype is not possible, check that NumPy raises a TypeError
            else:
                with pytest.raises(TypeError):
                    ufunc(pc1.data, pc2.data)
                with pytest.raises(TypeError):
                    ufunc(pc1, pc2)

    @pytest.mark.parametrize("arrfunc_str", handled_functions_1in)  # type: ignore
    @pytest.mark.parametrize("dtype", ["uint8", "int8", "float32"])  # type: ignore
    def test_array_functions_1nin(self, arrfunc_str: str, dtype: str) -> None:
        """
        Test that single-input array functions that we support give the same output as they would on the masked array.
        """

        data = self.data1.astype(dtype)
        # If floating type, create NaNs
        if np.issubdtype(dtype, np.floating):
            data[self.mask1] = np.nan
        pc = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data, crs=self.crs)

        # Get array func
        arrfunc = getattr(np, arrfunc_str)

        # Find the common dtype between the point cloud and the most constrained input type (first character is input)
        # com_dtype = np.find_common_type([dtype] + [arrfunc.types[0][0]], [])

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            if "percentile" in arrfunc_str:
                args = (80.0,)
            elif "quantile" in arrfunc_str:
                args = (0.8,)
            else:
                args = ()  # type: ignore

            output_pc = arrfunc(pc, *args)
            output_arr = arrfunc(pc.data, *args)

            # This test is for when the NumPy function reduces the dimension of the array but not completely
            if isinstance(output_arr, np.ndarray):
                # When the NumPy function preserves the shape, it returns a point cloud
                if output_arr.shape == pc.data.shape:
                    assert isinstance(output_pc, gu.PointCloud)
                    assert np.array_equal(output_pc.data, output_arr, equal_nan=True)
                # Otherwise, it returns an array
                else:
                    assert np.array_equal(output_pc, output_arr, equal_nan=True)
            # This test is for when the NumPy function reduces the dimension to a single number
            else:
                assert np.array_equal(output_pc, output_arr, equal_nan=True)

    @pytest.mark.parametrize("arrfunc_str", handled_functions_2in)  # type: ignore
    @pytest.mark.parametrize("dtype1", ["uint8", "int8", "float32"])  # type: ignore
    @pytest.mark.parametrize("dtype2", ["uint8", "int8", "float32"])  # type: ignore
    def test_array_functions_2nin(self, arrfunc_str: str, dtype1: str, dtype2: str) -> None:
        """
        Test that double-input array functions that we support give the same output as they would on the masked array.
        """

        data1 = self.data1.astype(dtype1)
        data2 = self.data2.astype(dtype2)
        # If floating type, create NaNs
        if np.issubdtype(dtype1, np.floating):
            data1[self.mask1] = np.nan
        if np.issubdtype(dtype2, np.floating):
            data2[self.mask2] = np.nan

        pc1 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data1, crs=self.crs)
        pc2 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data2, crs=self.crs)

        # Get array func
        arrfunc = getattr(np, arrfunc_str)

        # Catch warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            # Compute outputs
            output_pc = arrfunc(pc1, pc2)
            output_arr = arrfunc(pc1.data, pc2.data)

            # When the NumPy function preserves the shape, it returns a point cloud
            if isinstance(output_arr, np.ndarray) and output_arr.shape == pc1.data.shape:
                assert isinstance(output_pc, gu.PointCloud)
                assert np.array_equal(output_pc.data, output_arr, equal_nan=True)
            # Otherwise, it returns an array
            else:
                assert np.array_equal(output_pc, output_arr, equal_nan=True)

    @pytest.mark.parametrize("method_str", ["reduce"])  # type: ignore
    def test_ufunc_methods(self, method_str):
        """
        Test that universal function methods all behave properly, don't need to test all
        nodatas and dtypes as this was done above.
        """

        data2 = self.data2.astype("float32")
        data3 = self.data3.astype("float32")
        data1 = self.data1.astype("float32")
        data1[self.mask1] = np.nan
        data2[self.mask2] = np.nan
        data3[self.mask3] = np.nan

        pc1 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data1, crs=self.crs)
        pc2 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data2, crs=self.crs)
        pc3 = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=data3, crs=self.crs)

        # Methods reduce, accumulate, reduceat and at only supported for binary function (2nin)
        # -- Test 1: -- Try a ufunc with 2nin, 1nout like np.add
        ufunc_2nin_1nout = getattr(np.add, method_str)
        output_pc = ufunc_2nin_1nout((pc1, pc2, pc3))
        output_arr = ufunc_2nin_1nout((pc1.data, pc2.data, pc3.data))

        assert np.array_equal(output_pc.data, output_arr, equal_nan=True)

        # Methods reduce only supports function that output a single value
        # -- Test 2: -- Try a ufunc with 2nin, 2nout: there's only divmod
        # ufunc_2nin_2nout = getattr(np.divmod, method_str)
        # outputs_pc = ufunc_2nin_2nout((rst1, rst2, rst3))
        # outputs_ma = ufunc_2nin_2nout((ma1, ma2, ma3))
        #
        # assert np.ma.allequal(outputs_ma[0], outputs_pc[0].data) and np.ma.allequal(
        #             outputs_ma[1], outputs_pc[1].data)

    @pytest.mark.parametrize(
        "np_func_name", ufuncs_str_2nin_1nout + ufuncs_str_2nin_2nout + handled_functions_2in
    )  # type: ignore
    def test_raise_errors_2nin(self, np_func_name: str) -> None:
        """Check that proper errors are raised when input pointcloud/array don't match (only 2-input functions)."""

        # Create point clouds
        pc = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=self.data1, crs=self.crs)
        pc_wrong_shape = gu.PointCloud.from_array(self.arr_wrong_shape, crs=self.crs)
        pc_wrong_crs = gu.PointCloud.from_xyz(x=self.coords[0], y=self.coords[1], z=self.data1, crs=self.wrong_crs)
        pc_wrong_coords = gu.PointCloud.from_array(self.arr_wrong_coord, crs=self.crs)

        # Get ufunc
        np_func = getattr(np, np_func_name)

        # Strange errors happening only for these 4 functions...
        # See issue #457
        if np_func_name not in ["allclose", "isclose", "array_equal", "array_equiv"]:

            # Point clouds with different CRS or shape
            # Different shape
            georef_twopc_message = (
                "Both point clouds must have the same points X/Y coordinates and CRS for an arithmetic operation."
            )
            with pytest.raises(ValueError, match=re.escape(georef_twopc_message)):
                np_func(pc, pc_wrong_shape)

            with pytest.raises(ValueError, match=re.escape(georef_twopc_message)):
                np_func(pc, pc_wrong_coords)

            # Different CRS
            with pytest.raises(ValueError, match=re.escape(georef_twopc_message)):
                np_func(pc, pc_wrong_crs)

            # Array with different shape
            georef_pc_array_message = (
                "The array must be 1-dimensional with the same number of points as the point cloud for an arithmetic "
                "operation."
            )
            # Different shape, masked array
            # Check reflectivity just in case (just here, not later)
            with pytest.raises(ValueError, match=re.escape(georef_pc_array_message)):
                np_func(pc_wrong_shape.data, pc)
            with pytest.raises(ValueError, match=re.escape(georef_pc_array_message)):
                np_func(pc, pc_wrong_shape.data)
