"""Tests on Pandas accessor mirroring PointCloud API."""

from __future__ import annotations

import os.path
import tempfile
import warnings
from importlib.util import find_spec

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal

import geoutils as gu
import geoutils.pointcloud.pd_accessor as pd_accessor
from geoutils.multiproc import MultiprocConfig


class TestPointCloudAccessor:
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(20, 3)) + rng.normal(0, 0.15, size=(20, 3))
    gdf = gpd.GeoDataFrame(
        data={"z": arr_points[:, 2]},
        geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1]),
        crs=4326,
    )
    fn_las = gu.examples.get_path_test("coromandel_lidar")

    def test_accessor(self) -> None:
        ds = self.gdf.copy()

        assert ds.pc.data_column == "z"
        assert ds.pc.point_count == len(ds)
        assert np.array_equal(ds.pc.data.values, ds["z"].values)
        assert isinstance(ds.pc.to_geoutils(), gu.PointCloud)

    def test_copy_and_arithmetic(self) -> None:
        ds = self.gdf.copy()

        copied = ds.pc.copy()
        assert_geodataframe_equal(copied, ds)

        summed = ds.pc + 1
        assert isinstance(summed, gpd.GeoDataFrame)
        assert np.array_equal(summed["z"].values, ds["z"].values + 1)

    def test_from_xyz(self) -> None:
        ds = gu.PointCloudAccessor.from_xyz(
            x=self.arr_points[:, 0],
            y=self.arr_points[:, 1],
            z=self.arr_points[:, 2],
            crs=4326,
            data_column="z",
        )

        assert isinstance(ds, gpd.GeoDataFrame)
        assert ds.pc.to_geoutils().pointcloud_equal(gu.PointCloud(self.gdf, data_column="z"))

    def test_cross_type_outputs_are_accessors(self) -> None:
        ds = gu.PointCloudAccessor.from_xyz(
            x=np.array([0, 1, 0, 1]),
            y=np.array([0, 0, 1, 1]),
            z=np.array([1, 2, 3, 4]),
            crs=3857,
            data_column="z",
        )

        raster = ds.pc.grid(
            grid_coords=(np.array([0, 1]), np.array([0, 1])),
            resampling="nearest",
            dist_nodata_pixel=10,
        )

        assert isinstance(raster, xr.DataArray)

    def test_open_pointcloud(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="z")

        assert isinstance(ds, gpd.GeoDataFrame)
        assert ds.pc.to_geoutils().pointcloud_equal(gu.PointCloud(self.gdf, data_column="z"))

    def test_open_pointcloud__dask(self) -> None:
        pytest.importorskip("dask")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.dataframe as dd

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5)

        assert isinstance(ds, dd.DataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == len(self.gdf)

        summed = ds.pc + 1
        assert isinstance(summed, dd.DataFrame)
        assert np.array_equal(summed.compute()["z"].values, self.gdf["z"].values + 1)

    def test_open_pointcloud__dask_missing_dep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)
        import_optional = pd_accessor.import_optional

        def _missing_dask(import_name: str, package_name: str | None = None, extra_name: str = "opt") -> object:
            if import_name == "dask":
                raise ImportError("Optional dependency 'dask' required.")
            return import_optional(import_name, package_name=package_name, extra_name=extra_name)

        monkeypatch.setattr(pd_accessor, "import_optional", _missing_dask)
        with pytest.raises(ImportError, match="Optional dependency 'dask' required.*"):
            gu.open_pointcloud(temp_file, data_column="z", chunks=5)

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_open_pointcloud_las__dask(self) -> None:
        pytest.importorskip("dask")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="dask.dataframe")
            import dask.dataframe as dd

        ds = gu.open_pointcloud(self.fn_las, chunks=100)
        pc = gu.PointCloud(self.fn_las)

        assert isinstance(ds, dd.DataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == pc.point_count
        assert len(ds.compute()) == pc.point_count

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_load_las__multiprocessing(self) -> None:
        pc_chunked = gu.PointCloud(self.fn_las)
        pc_chunked.load(mp_config=MultiprocConfig(chunks=100))

        pc = gu.PointCloud(self.fn_las)
        pc.load()

        assert pc_chunked.pointcloud_equal(pc)
