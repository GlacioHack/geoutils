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
import xarray as xr
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
        if isinstance(output_ds, PointCloud):
            assert output_pc.pointcloud_equal(output_ds)
        else:
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

    # For tabular statistics
    elif isinstance(output_pc, pd.DataFrame):
        assert_frame_equal(output_pc, output_ds)

    # For lightweight variogram records
    elif isinstance(output_pc, gu.Variogram):
        assert isinstance(output_ds, gu.Variogram)
        assert np.allclose(output_pc.lags, output_ds.lags)
        assert np.allclose(output_pc.semivariance, output_ds.semivariance, equal_nan=True)
        assert np.array_equal(output_pc.counts, output_ds.counts)
        assert output_pc.model == output_ds.model

    # For bounded cosampling results
    elif isinstance(output_pc, gu.CoSampleResult):
        assert isinstance(output_ds, gu.CoSampleResult)
        assert np.array_equal(output_pc.self_values, output_ds.self_values)
        assert np.array_equal(output_pc.other_values, output_ds.other_values)
        assert np.array_equal(output_pc.indices, output_ds.indices)

    # For labelled pair samples
    elif isinstance(output_pc, xr.Dataset):
        assert output_pc.identical(output_ds)

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
        ("pointcloud_allclose", {"other": "self"}),
        ("georeferenced_coords_equal", {"pc": "self"}),
        ("get_stats", {}),
        ("grouped_stats", {"by": {"group": "b2"}, "bins": {"group": 2}, "statistics": "mean"}),
        ("subsample", {"subsample": 2, "random_state": 42}),
        ("cosample", {"other": "self", "subsample": 2, "random_state": 42}),
        (
            "sample_pairs",
            {"n_pairs": 4, "min_distance": 0.5, "max_distance": 2, "strategy": "kdtree", "random_state": 42},
        ),
        (
            "variogram",
            {
                "n_pairs": 4,
                "n_lags": 2,
                "min_lag": 0.5,
                "max_lag": 2,
                "strategy": "kdtree",
                "random_state": 42,
            },
        ),
        ("to_geoutils", {}),
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
        if method == "variogram":
            pytest.importorskip("skgstat")

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

    def test_equality__cross_type_and_tolerance(self) -> None:
        """Check that equality accepts both APIs while allclose tolerates small numeric differences."""

        pointcloud = PointCloud(self.ds, data_column="b1")
        exact_ds = self.ds.copy()
        exact_ds.pc.set_data_column("b1")
        close_ds = self.ds.copy()
        close_ds.geometry = close_ds.geometry.translate(xoff=1e-9)
        close_ds["b1"] += 1e-9
        close_ds.pc.set_data_column("b1")

        assert pointcloud.pointcloud_equal(exact_ds.pc)
        assert exact_ds.pc.pointcloud_equal(pointcloud)
        assert not pointcloud.pointcloud_equal(close_ds)
        assert pointcloud.pointcloud_allclose(close_ds, atol=1e-8)
        assert close_ds.pc.pointcloud_allclose(pointcloud, atol=1e-8)
        assert not pointcloud.pointcloud_allclose(close_ds, rtol=0, atol=1e-10)

    def test_copy__preserves_dataframe_and_pointcloud_type(self) -> None:
        """Check that copying retains auxiliary columns, indexes and PointCloud outputs."""

        ds = self.ds.copy()
        ds.index = pd.Index([10, 20, 30, 40], name="point_id")
        pointcloud = PointCloud(ds, data_column="b1")
        replacement = np.array([11.0, 12.0, 13.0, 14.0])

        copied = pointcloud.copy(new_array=replacement)
        copied_ds = ds.pc.copy(new_array=replacement)

        assert isinstance(copied, PointCloud)
        assert copied.columns.equals(ds.columns)
        assert copied.index.equals(ds.index)
        assert np.array_equal(copied["b1"], replacement)
        assert np.array_equal(copied["b2"], ds["b2"])
        assert_geodataframe_equal(copied.ds, copied_ds)
        assert np.array_equal(pointcloud["b1"], ds["b1"])

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("crop", {"bbox": (-1, -1, 0.5, 2)}),
            ("reproject", {"crs": 4326}),
            ("translate", {"xoff": 1, "yoff": 2}),
        ],
    )
    def test_point_preserving_vector_methods__return_pointcloud(self, method: str, kwargs: dict[str, Any]) -> None:
        """Check that inherited point-preserving vector methods retain PointCloud semantics."""

        pointcloud = PointCloud(self.ds, data_column="b1")
        result = getattr(pointcloud, method)(**kwargs)

        assert isinstance(result, PointCloud)
        assert result.data_column == "b1"
        assert "b2" in result.columns
        assert isinstance(pointcloud.to_geoutils(), PointCloud)

    def test_shared_methods_and_arithmetic_ownership(self) -> None:
        """Check that shared operations live in the base while arithmetic remains exclusive to PointCloud."""

        shared_methods = {"from_xyz", "pointcloud_equal", "pointcloud_allclose", "get_stats", "grid"}
        assert shared_methods <= set(PointCloudBase.__dict__)
        assert shared_methods.isdisjoint(PointCloud.__dict__)
        assert "__add__" not in PointCloudBase.__dict__
        assert "__add__" in PointCloud.__dict__

        with pytest.raises(TypeError):
            self.ds.pc + 1


class TestAccessorDask:
    """Test Dask loading and laziness of the "pc" Pandas accessor."""

    # Use the same compact fixture as the eager class-versus-accessor tests
    ds = TestClassVsAccessorConsistency.ds

    def test_open__dask(self) -> None:
        """
        Check that a DataFrame opened with chunks using "open_pointcloud" maintains Dask laziness.
        """

        # Skip cleanly when the optional lazy dataframe backend is unavailable
        dgpd = pytest.importorskip("dask_geopandas")

        # Write a source that can be reopened into two row partitions
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        # Metadata queries should not force the Dask collection into memory
        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)

        assert isinstance(ds, dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == len(self.ds)

    def test_chunked_methods__equality_loading_laziness(self) -> None:
        """
        Test that chunked methods have the exact same output, loading mechanism and laziness.
        """

        # Load both lazy dataframe and lazy array types used by this test
        pytest.importorskip("dask_geopandas")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.array as da

        # Prepare matching lazy and eager point-cloud interfaces
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)
        pc = PointCloud(self.ds, data_column="b1")

        # Array conversion should create a Dask array with the eager values
        array_ds = ds.pc.to_array()
        array_pc = pc.to_array()

        assert isinstance(array_ds, da.Array)
        assert np.array_equal(array_ds.compute(), array_pc)
        assert not ds.pc.is_loaded

        # Loading returns an eager replacement because a Dask collection cannot be mutated in place
        loaded = ds.pc.load()
        assert_geodataframe_equal(loaded, self.ds)
        assert not ds.pc.is_loaded

    def test_chunked_equality__compares_values_without_loading(self) -> None:
        """Check that lazy equality detects changed coordinates and values without replacing either input."""

        pytest.importorskip("dask_geopandas")

        with tempfile.TemporaryDirectory() as temp_dir:
            source = os.path.join(temp_dir, "source.gpkg")
            changed = os.path.join(temp_dir, "changed.gpkg")
            self.ds.to_file(source)
            changed_ds = self.ds.copy()
            changed_ds.geometry = changed_ds.geometry.translate(xoff=10)
            changed_ds.to_file(changed)

            lazy = gu.open_pointcloud(source, data_column="b1", chunks=3)
            lazy_changed = gu.open_pointcloud(changed, data_column="b1", chunks=2)
            eager = PointCloud(self.ds, data_column="b1")

            assert lazy.pc.pointcloud_equal(eager)
            assert lazy.pc.georeferenced_coords_equal(eager)
            assert not lazy_changed.pc.georeferenced_coords_equal(eager)
            assert not lazy_changed.pc.pointcloud_equal(eager)
            assert not lazy.pc.is_loaded
            assert not lazy_changed.pc.is_loaded

    def test_chunked_reduction_methods__equality_loading_laziness(self) -> None:
        """Test Dask point-cloud reductions/subsampling compute small outputs without loading the source."""

        pytest.importorskip("dask_geopandas")

        # Open the same source lazily and eagerly for reduction comparisons
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)
        pc = PointCloud(self.ds, data_column="b1")

        # Statistics compute a small dictionary without loading the accessor source
        assert_output_equal(
            pc.get_stats(["mean", "max", "valid_count"]),
            ds.pc.get_stats(["mean", "max", "valid_count"]),
        )
        assert not ds.pc.is_loaded

        # Subsampling computes only the requested small point selection
        assert_output_equal(
            pc.subsample(subsample=2, random_state=42),
            ds.pc.subsample(subsample=2, random_state=42),
        )
        assert not ds.pc.is_loaded

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_open_las__multiprocessing_and_dask(self) -> None:
        """Check LAS chunked loading through Multiprocessing and Dask."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Establish one eager reference for both chunked backends
        fn_las = gu.examples.get_path_test("coromandel_lidar")

        pc = PointCloud(fn_las)
        pc.load()

        # Multiprocessing reads independent LAS row chunks into one PointCloud
        pc_mp = PointCloud(fn_las)
        pc_mp.load(mp_config=MultiprocConfig(chunks=100))

        assert pc_mp.is_loaded
        assert pc_mp.pointcloud_equal(pc)

        # Dask represents the same LAS chunks as a lazy GeoDataFrame
        ds = gu.open_pointcloud(fn_las, chunks=100)
        assert isinstance(ds, dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == pc.point_count

        # Compute once for a complete coordinate and value comparison
        ds_comp = _as_geodataframe(ds.compute(), crs=ds.pc.crs)
        ds_pc = PointCloud(ds_comp, data_column="Z")
        assert ds_pc.georeferenced_coords_equal(pc)
        assert np.allclose(ds_pc.data, pc.data)
        assert not ds.pc.is_loaded

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_grid_las__multiprocessing_and_dask(self) -> None:
        """Test LAS point-cloud gridding with Dask and Multiprocessing."""

        pytest.importorskip("dask_geopandas")
        import dask.array as da

        # Build a small eager grid that both chunked implementations must match
        fn_las = gu.examples.get_path_test("coromandel_lidar")

        pc = PointCloud(fn_las)
        pc.load()
        expected = pc.grid(
            shape=(3, 3),
            bounds=pc.bounds,
            resampling="nearest",
            dist_nodata_pixel=100,
        )

        # Dask selects and grids point partitions separately for every output block
        ds = gu.open_pointcloud(fn_las, chunks=100)
        output_dask = ds.pc.grid(
            shape=(3, 3),
            bounds=pc.bounds,
            resampling="nearest",
            dist_nodata_pixel=100,
            chunksizes=(2, 1),
        )

        # Check graph layout and laziness before computing the grid
        assert isinstance(output_dask.data, da.Array)
        assert output_dask.data.chunks == ((2, 1), (1, 1, 1))
        assert not ds.pc.is_loaded
        assert np.array_equal(expected.data, output_dask.compute().values, equal_nan=True)
        assert not ds.pc.is_loaded
        assert not output_dask._in_memory

        # Multiprocessing keeps the PointCloud unloaded while workers read LAS bounds
        pc_file = PointCloud(fn_las)
        output_mp = pc_file.grid(
            shape=(3, 3),
            bounds=pc.bounds,
            resampling="nearest",
            dist_nodata_pixel=100,
            mp_config=MultiprocConfig(chunks=(2, 1)),
        )

        # Its file-backed raster result should equal the eager and Dask results
        assert not pc_file.is_loaded
        assert not output_mp.is_loaded
        assert np.array_equal(expected.data, output_mp.data, equal_nan=True)
        assert not pc_file.is_loaded
        assert output_mp.is_loaded
