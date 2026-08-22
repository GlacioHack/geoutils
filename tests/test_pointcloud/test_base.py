"""Test PointCloudBase class, parent of PointCloud class and 'pc' Pandas accessor."""

from __future__ import annotations

import os.path
import tempfile
import warnings
from importlib.util import find_spec
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from pyproj import CRS

import geoutils as gu
from geoutils import PointCloud, Raster
from geoutils.multiproc import MultiprocConfig
from geoutils.pointcloud.base import PointCloudBase
from geoutils.pointcloud.pd_accessor import PointCloudAccessor


def _as_geodataframe(ds: gpd.GeoDataFrame | pd.DataFrame, crs: CRS | None = None) -> gpd.GeoDataFrame:
    """Convert a DataFrame with geometry objects to a GeoDataFrame."""

    if isinstance(ds, gpd.GeoDataFrame):
        return ds
    return gpd.GeoDataFrame(ds, geometry="geometry", crs=crs)


def assert_output_equal(output_pc: Any, output_ds: Any, use_allclose: bool = False) -> None:
    """Return equality of different output types."""

    # For point clouds: the class returns a PointCloud, while the accessor returns a GeoDataFrame
    if isinstance(output_pc, PointCloud):
        assert isinstance(output_ds, gpd.GeoDataFrame)
        assert output_pc.pointcloud_equal(PointCloud(output_ds, data_column=output_ds.pc.data_column))

    # For rasters
    elif isinstance(output_pc, Raster):
        if use_allclose:
            assert output_pc.raster_allclose(output_ds, warn_failure_reason=True, strict_masked=False)
        else:
            assert output_pc.raster_equal(output_ds, warn_failure_reason=True, strict_masked=False)

    # For arrays
    elif isinstance(output_pc, np.ndarray):
        if use_allclose:
            assert np.allclose(output_pc, np.asarray(output_ds), equal_nan=True)
        else:
            assert np.array_equal(output_pc, np.asarray(output_ds), equal_nan=True)

    # For tuple of arrays
    elif isinstance(output_pc, tuple) and isinstance(output_pc[0], np.ndarray):
        assert np.array_equal(np.array(output_pc), np.array([np.asarray(o) for o in output_ds]), equal_nan=True)

    # For a dictionary of numeric values
    elif isinstance(output_pc, dict):
        df1 = pd.DataFrame(index=[0], data=output_pc)
        df2 = pd.DataFrame(index=[0], data=output_ds)
        assert_frame_equal(df1, df2, check_dtype=False)

    # For GeoPandas objects
    elif isinstance(output_pc, gpd.GeoDataFrame):
        assert_geodataframe_equal(output_pc, output_ds)

    # For any other object type
    else:
        assert output_pc == output_ds


class NeedsTestError(ValueError):
    """Error to remember to add test when a new PointCloudBase method is added."""


class TestClassVsAccessorConsistency:
    """
    Test class to check the consistency between the outputs and loading of the PointCloud class and Pandas accessor.
    """

    ds = gpd.GeoDataFrame(
        data={"b1": np.array([1.0, 2.0, 3.0, 4.0]), "b2": np.array([5.0, 6.0, 7.0, 8.0])},
        geometry=gpd.points_from_xy(x=np.array([0.0, 1.0, 0.0, 1.0]), y=np.array([0.0, 0.0, 1.0, 1.0])),
        crs=CRS.from_epsg(32610),
    )

    # Get all PointCloudBase public properties and methods, ensures we test everything even with API changes
    properties = [k for k, v in PointCloudBase.__dict__.items() if not k.startswith("_") and isinstance(v, property)]
    methods = [k for k, v in PointCloudBase.__dict__.items() if not k.startswith("_") and not isinstance(v, property)]

    @pytest.mark.parametrize("prop", properties)
    def test_properties__equality_and_loading(self, prop: str) -> None:
        """
        Test that properties are exactly equal between a PointCloud and a GeoDataFrame using the "pc" accessor.
        """

        pc = PointCloud(self.ds, data_column="b1")
        ds = self.ds.copy()
        ds.pc.set_data_column("b1")

        output_pc = getattr(pc, prop)
        output_ds = getattr(ds.pc, prop)

        assert_output_equal(output_pc, output_ds)
        assert pc.is_loaded
        assert ds.pc.is_loaded

    methods_and_kwargs = [
        ("set_data_column", {"new_data_column": "b2"}),
        ("copy", {}),
        ("to_xyz", {}),
        ("to_array", {}),
        ("to_tuples", {}),
        ("pointcloud_equal", {"other": "self"}),
        ("georeferenced_coords_equal", {"pc": "self"}),
        ("get_stats", {}),
        ("subsample", {"subsample": 2, "random_state": 42}),
        (
            "grid",
            {"grid_coords": (np.array([0.0, 1.0]), np.array([0.0, 1.0])), "resampling": "nearest"},
        ),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in methods_and_kwargs])
    def test_methods__equality_and_loading(self, method: str, kwargs: dict[str, Any]) -> None:
        """
        Test that method output and loading are the same between a PointCloud and a GeoDataFrame "pc" accessor.
        """

        pc = PointCloud(self.ds, data_column="b1")
        ds = self.ds.copy()
        ds.pc.set_data_column("b1")

        args_pc = kwargs.copy()
        args_ds = kwargs.copy()
        if args_pc.get("other") == "self":
            args_pc["other"] = pc
            args_ds["other"] = ds
        if args_pc.get("pc") == "self":
            args_pc["pc"] = pc
            args_ds["pc"] = ds

        output_pc = getattr(pc, method)(**args_pc)
        output_ds = getattr(ds.pc, method)(**args_ds)

        if method == "set_data_column":
            assert output_pc is None
            assert output_ds is None
            assert pc.data_column == ds.pc.data_column
        else:
            assert_output_equal(output_pc, output_ds, use_allclose=method == "grid")

        assert pc.is_loaded
        assert ds.pc.is_loaded

    class_methods_and_kwargs = [
        (
            "from_xyz",
            {
                "x": ds.geometry.x.values,
                "y": ds.geometry.y.values,
                "z": ds["b1"].values,
                "crs": CRS.from_epsg(32610),
                "data_column": "b1",
            },
        ),
        (
            "from_array",
            {
                "data": np.vstack((ds.geometry.x.values, ds.geometry.y.values, ds["b1"].values)),
                "crs": CRS.from_epsg(32610),
                "data_column": "b1",
            },
        ),
        (
            "from_tuples",
            {
                "tuples_xyz": list(zip(ds.geometry.x.values, ds.geometry.y.values, ds["b1"].values)),
                "crs": CRS.from_epsg(32610),
                "data_column": "b1",
            },
        ),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in class_methods_and_kwargs])
    def test_classmethods__equality(self, method: str, kwargs: dict[str, Any]) -> None:
        """Test class method output exactly the same objects."""

        output_pc = getattr(PointCloud, method)(**kwargs)
        output_ds = getattr(PointCloudAccessor, method)(**kwargs)

        assert_output_equal(output_pc, output_ds)

    def test_methods__test_coverage(self) -> None:
        """Test that checks that all existing PointCloudBase methods are tested above."""

        methods_1 = [m[0] for m in self.methods_and_kwargs]
        methods_2 = [m[0] for m in self.class_methods_and_kwargs]
        list_missing = [method for method in self.methods if method not in methods_1 + methods_2]

        if len(list_missing) != 0:
            raise NeedsTestError(f"PointCloudBase methods not covered by tests: {list_missing}")


class TestAccessorDask:
    """Test Dask loading and laziness of the "pc" Pandas accessor."""

    ds = TestClassVsAccessorConsistency.ds

    def test_open__dask(self) -> None:
        """
        Check that a DataFrame opened with chunks using "open_pointcloud" maintains Dask laziness.
        """

        pytest.importorskip("dask")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.dataframe as dd

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)

        assert isinstance(ds, dd.DataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == len(self.ds)

    def test_chunked_methods__equality_loading_laziness(self) -> None:
        """
        Test that chunked methods have the exact same output, loading mechanism and laziness.
        """

        pytest.importorskip("dask")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.array as da
            import dask.dataframe as dd

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)
        pc = PointCloud(self.ds, data_column="b1")

        output_ds = ds.pc + 1
        output_pc = pc + 1

        assert not ds.pc.is_loaded
        assert isinstance(output_ds, dd.DataFrame)
        assert output_pc.is_loaded

        output_ds_comp = _as_geodataframe(output_ds.compute(), crs=ds.pc.crs)
        output_ds_pc = PointCloud(output_ds_comp, data_column="b1")
        assert output_pc.georeferenced_coords_equal(output_ds_pc)
        assert np.array_equal(output_pc.data, output_ds_pc.data)

        array_ds = ds.pc.to_array()
        array_pc = pc.to_array()

        assert isinstance(array_ds, da.Array)
        assert np.array_equal(array_ds.compute(), array_pc)

        ds.pc.load()
        assert ds.pc.is_loaded
        assert_geodataframe_equal(ds.pc.ds, self.ds)

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_open_las__multiprocessing_and_dask(self) -> None:
        """Check LAS chunked loading through Multiprocessing and Dask."""

        pytest.importorskip("dask")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.dataframe as dd

        fn_las = gu.examples.get_path_test("coromandel_lidar")

        pc = PointCloud(fn_las)
        pc.load()

        pc_mp = PointCloud(fn_las)
        pc_mp.load(mp_config=MultiprocConfig(chunks=100))

        assert pc_mp.pointcloud_equal(pc)

        ds = gu.open_pointcloud(fn_las, chunks=100)
        assert isinstance(ds, dd.DataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == pc.point_count

        ds_comp = _as_geodataframe(ds.compute(), crs=ds.pc.crs)
        ds_pc = PointCloud(ds_comp, data_column="Z")
        assert ds_pc.georeferenced_coords_equal(pc)
        assert np.allclose(ds_pc.data, pc.data)
