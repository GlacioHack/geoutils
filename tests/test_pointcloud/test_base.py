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
from geoutils.interface.gridding import GriddingMethod
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
        dgpd = pytest.importorskip("dask_geopandas")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.array as da

        # Prepare matching lazy and eager point-cloud interfaces
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)
        pc = PointCloud(self.ds, data_column="b1")

        # Arithmetic should remain lazy only for the Dask-backed accessor
        output_ds = ds.pc + 1
        output_pc = pc + 1

        assert not ds.pc.is_loaded
        assert isinstance(output_ds, dgpd.GeoDataFrame)
        assert output_pc.is_loaded

        # Compute the accessor result and compare both coordinates and values
        output_ds_comp = _as_geodataframe(output_ds.compute(), crs=ds.pc.crs)
        output_ds_pc = PointCloud(output_ds_comp, data_column="b1")
        assert output_pc.georeferenced_coords_equal(output_ds_pc)
        assert np.array_equal(output_pc.data, output_ds_pc.data)

        # Array conversion should create a Dask array with the eager values
        array_ds = ds.pc.to_array()
        array_pc = pc.to_array()

        assert isinstance(array_ds, da.Array)
        assert np.array_equal(array_ds.compute(), array_pc)

        # Explicit loading is the only operation that replaces the lazy source
        ds.pc.load()
        assert ds.pc.is_loaded
        assert_geodataframe_equal(ds.pc.ds, self.ds)

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

    @pytest.mark.parametrize(
        "resampling",
        [
            "nearest",
            "idw",
            "mean",
            "minimum",
            "maximum",
            "range",
            "count",
            "stdev",
            "average_distance",
            "average_distance_pts",
        ],
    )
    def test_grid__dask_geopandas(self, resampling: GriddingMethod) -> None:
        """Test interpolation and neighborhood gridding from a Dask-GeoPandas input."""

        pytest.importorskip("dask_geopandas")
        import dask.array as da

        # Build an eager reference before opening the partitioned source
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        expected = PointCloud(self.ds, data_column="b1").grid(
            res=1,
            bounds=(0, 0, 2, 2),
            resampling=resampling,
            dist_nodata_pixel=2,
        )
        # Request rectangular output chunks to exercise Dask block assembly
        ds = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)
        output = ds.pc.grid(
            res=1,
            bounds=(0, 0, 2, 2),
            resampling=resampling,
            dist_nodata_pixel=2,
            chunksizes=(2, 1),
        )

        # Verify lazy output structure before computing numerical equality
        assert isinstance(output.data, da.Array)
        assert output.data.chunks == ((2,), (1, 1))
        assert not ds.pc.is_loaded
        assert np.array_equal(expected.data, output.compute().values, equal_nan=True)
        assert not ds.pc.is_loaded

    @pytest.mark.parametrize(
        "resampling",
        [
            "nearest",
            "idw",
            "mean",
            "minimum",
            "maximum",
            "range",
            "count",
            "stdev",
            "average_distance",
            "average_distance_pts",
        ],
    )
    def test_grid__multiprocessing_file_backed(self, resampling: GriddingMethod) -> None:
        """Test interpolation and neighborhood gridding from an unloaded file with Multiprocessing."""

        # Build the expected grid from the complete in-memory point cloud
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)

        expected = PointCloud(self.ds, data_column="b1").grid(
            res=1,
            bounds=(0, 0, 2, 2),
            resampling=resampling,
            dist_nodata_pixel=2,
        )
        # Keep the source unloaded while worker tasks read only their point bounds
        pc = PointCloud(temp_file, data_column="b1")
        output = pc.grid(
            res=1,
            bounds=(0, 0, 2, 2),
            resampling=resampling,
            dist_nodata_pixel=2,
            mp_config=MultiprocConfig(chunks=(2, 1)),
        )

        # The result stays file-backed and matches the eager reference when read
        assert not pc.is_loaded
        assert not output.is_loaded
        assert np.array_equal(expected.data, output.data, equal_nan=True)

    @pytest.mark.parametrize("resampling", ["nearest", "mean"])
    def test_grid__numba_backends(self, resampling: GriddingMethod) -> None:
        """Keep the Numba calculation engine identical across eager, Dask and Multiprocessing backends."""

        pytest.importorskip("numba")
        pytest.importorskip("dask_geopandas")

        # Store one point source so both out-of-core backends can select local points
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.ds.to_file(temp_file)
        kwargs = {
            "res": 1,
            "bounds": (0, 0, 2, 2),
            "resampling": resampling,
            "dist_nodata_pixel": 2,
            "engine": "numba",
        }

        # The eager result provides the complete reference for both tiled outputs
        expected = PointCloud(self.ds, data_column="b1").grid(**kwargs)
        dask_points = gu.open_pointcloud(temp_file, data_column="b1", chunks=2)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 1))
        assert np.array_equal(expected.data, dask_output.compute().values, equal_nan=True)

        # Multiprocessing forwards the same calculation engine to each worker tile
        file_points = PointCloud(temp_file, data_column="b1")
        multiproc_output = file_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 1)),
        )
        assert np.array_equal(expected.data, multiproc_output.data, equal_nan=True)

    def test_grid__nodata_propagation_backends(self) -> None:
        """Keep propagated invalid points identical across eager, Dask and Multiprocessing gridding."""

        pytest.importorskip("dask_geopandas")

        # A regular grid with one invalid center exercises support across output chunk boundaries
        x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
        values = np.arange(9, dtype=float)
        values[4] = np.nan
        points = gpd.GeoDataFrame(
            data={"z": values},
            geometry=gpd.points_from_xy(x=x.ravel(), y=y.ravel()),
            crs=32610,
        )
        temp_dir = tempfile.TemporaryDirectory()
        point_file = os.path.join(temp_dir.name, "nodata-points.gpkg")
        points.to_file(point_file)
        grid_coords = (np.arange(3, dtype=float), np.arange(3, dtype=float))

        # The eager result establishes the complete support mask before the source is partitioned
        expected = PointCloud(points, data_column="z").grid(
            grid_coords=grid_coords,
            resampling="mean",
            dist_nodata_pixel=1.1,
            nodata_propagation="propagate",
        )
        dask_points = gu.open_pointcloud(point_file, data_column="z", chunks=2)
        dask_output = dask_points.pc.grid(
            grid_coords=grid_coords,
            resampling="mean",
            dist_nodata_pixel=1.1,
            nodata_propagation="propagate",
            chunksizes=(2, 2),
        )
        assert np.array_equal(expected.data, dask_output.compute().values, equal_nan=True)

        # Multiprocessing reads the same support around each output tile from the file-backed source
        file_points = PointCloud(point_file, data_column="z")
        multiproc_output = file_points.grid(
            grid_coords=grid_coords,
            resampling="mean",
            dist_nodata_pixel=1.1,
            nodata_propagation="propagate",
            mp_config=MultiprocConfig(chunks=(2, 2)),
        )
        assert np.array_equal(expected.data, multiproc_output.data, equal_nan=True)

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
