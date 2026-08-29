"""Tests on the Pandas accessor mirroring the PointCloud API."""

from __future__ import annotations

import os.path
import tempfile
from importlib.util import find_spec

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal

import geoutils as gu
import geoutils.vector.pd_accessor as vector_pd_accessor
from geoutils.multiproc import MultiprocConfig


class TestPointCloudAccessor:
    """Check that the Pandas ``pc`` accessor exposes PointCloud behavior and lazy Dask support."""

    # Reuse one deterministic point cloud across accessor construction and IO tests
    rng = np.random.default_rng(42)
    arr_points = rng.integers(low=1, high=1000, size=(20, 3)) + rng.normal(0, 0.15, size=(20, 3))
    gdf = gpd.GeoDataFrame(
        data={"z": arr_points[:, 2]},
        geometry=gpd.points_from_xy(x=arr_points[:, 0], y=arr_points[:, 1]),
        crs=4326,
    )
    fn_las = gu.examples.get_path_test("coromandel_lidar")

    def test_accessor(self) -> None:
        """Expose point-cloud metadata, values and conversion through the accessor."""

        # Accessing ``pc`` should infer the only non-geometry data column
        ds = self.gdf.copy()

        # Compare the lightweight accessor view with the underlying dataframe
        assert ds.pc.data_column == "z"
        assert ds.pc.point_count == len(ds)
        assert np.array_equal(ds.pc.data.values, ds["z"].values)
        assert isinstance(ds.pc.to_geoutils(), gu.PointCloud)

    def test_copy(self) -> None:
        """Return an independent GeoDataFrame from the accessor copy method."""

        ds = self.gdf.copy()

        # Copying must retain the complete geospatial dataframe
        copied = ds.pc.copy()
        assert_geodataframe_equal(copied, ds)

    def test_from_xyz(self) -> None:
        """Construct an accessor-backed GeoDataFrame directly from X/Y/Z arrays."""

        # Use the same coordinates and values as the shared expected point cloud
        ds = gu.PointCloudAccessor.from_xyz(
            x=self.arr_points[:, 0],
            y=self.arr_points[:, 1],
            z=self.arr_points[:, 2],
            crs=4326,
            data_column="z",
        )

        # Accessor construction should match construction through PointCloud
        assert isinstance(ds, gpd.GeoDataFrame)
        assert ds.pc.to_geoutils().pointcloud_equal(gu.PointCloud(self.gdf, data_column="z"))

    def test_cross_type_outputs_are_accessors(self) -> None:
        """Return an Xarray accessor when gridding changes the geospatial data type."""

        # Build four points covering a small regular raster grid
        ds = gu.PointCloudAccessor.from_xyz(
            x=np.array([0, 1, 0, 1]),
            y=np.array([0, 0, 1, 1]),
            z=np.array([1, 2, 3, 4]),
            crs=3857,
            data_column="z",
        )

        # The accessor API returns an Xarray object instead of a Raster wrapper
        raster = ds.pc.grid(
            grid_coords=(np.array([0, 1]), np.array([0, 1])),
            resampling="nearest",
            dist_nodata_pixel=10,
        )

        assert isinstance(raster, xr.DataArray)

    def test_open_pointcloud(self) -> None:
        """Open a vector file as a GeoDataFrame carrying the point-cloud accessor."""

        # Write a small independent source file for the public open helper
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        # The eager path should reproduce the original point cloud
        ds = gu.open_pointcloud(temp_file, data_column="z")

        assert isinstance(ds, gpd.GeoDataFrame)
        assert ds.pc.to_geoutils().pointcloud_equal(gu.PointCloud(self.gdf, data_column="z"))

    def test_open_pointcloud__dask(self) -> None:
        """Keep point-cloud opening lazy when chunks request Dask-GeoPandas."""

        # Skip cleanly when the optional lazy dataframe backend is unavailable
        dgpd = pytest.importorskip("dask_geopandas")

        # Write a source whose rows can be split into several Dask partitions
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        # Opening with chunks should expose metadata without reading every partition
        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5)

        assert isinstance(ds, dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == len(self.gdf)

    def test_reproject_pointcloud__dask_geopandas(self) -> None:
        """Reproject point partitions lazily and match eager GeoPandas output."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Open a partitioned source before requesting a new projected CRS
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5)
        reprojected = ds.pc.reproject(crs=3857)

        # Only the final comparison should compute the Dask collection
        assert isinstance(reprojected, dgpd.GeoDataFrame)
        assert not reprojected.pc.is_loaded
        assert_geodataframe_equal(reprojected.compute(), self.gdf.to_crs(3857))
        assert not ds.pc.is_loaded
        assert not reprojected.pc.is_loaded

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("copy", {}),
            ("crop", {"bbox": (0, 0, 500, 500)}),
            ("translate", {"xoff": 1, "yoff": 2}),
        ],
    )
    def test_geometric_methods__dask_geopandas(self, method: str, kwargs: dict[str, object]) -> None:
        """Keep copied, cropped and translated point partitions lazy and equal to eager GeoPandas."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Open one on-disk source lazily and build the expected result through the eager accessor
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)
        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5)
        expected = getattr(self.gdf.pc, method)(**kwargs)

        # Each dataframe operation should add work without evaluating any point partition
        output = getattr(ds.pc, method)(**kwargs)
        assert isinstance(output, dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert not output.pc.is_loaded
        assert_geodataframe_equal(output.compute(), expected)
        assert not ds.pc.is_loaded
        assert not output.pc.is_loaded

    def test_to_file__dask_geopandas(self) -> None:
        """Write a lazy point cloud to a regular GeoPandas-supported vector file."""

        pytest.importorskip("dask_geopandas")

        # Reproject lazily so writing must evaluate transformed partitions
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5).pc.reproject(crs=3857)
        assert not ds.pc.is_loaded
        output_file = os.path.join(temp_dir.name, "output.gpkg")
        ds.pc.to_file(output_file)

        # Reopen through PointCloud to validate geometry, CRS and data values
        assert os.path.exists(output_file)
        assert gu.PointCloud(output_file, data_column="z").pointcloud_equal(gu.PointCloud(self.gdf.to_crs(3857), "z"))
        assert not ds.pc.is_loaded

    def test_to_file__dask_geopandas_parquet(self) -> None:
        """Write Dask point partitions to one GeoParquet dataset without changing rows."""

        pytest.importorskip("dask_geopandas")
        pytest.importorskip("pyarrow")

        # Create a partitioned point cloud from a portable vector source
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)

        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5)
        assert not ds.pc.is_loaded
        output_file = os.path.join(temp_dir.name, "output.parquet")
        ds.pc.to_file(output_file)

        # Normalize indexes because dataset writers may rebuild them across partitions
        output = gpd.read_parquet(output_file).reset_index(drop=True)
        expected = self.gdf.reset_index(drop=True)
        assert os.path.exists(output_file)
        assert_geodataframe_equal(output, expected)
        assert not ds.pc.is_loaded

    def test_open_pointcloud__dask_missing_dep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explain the missing optional dependency when chunked opening requests Dask."""

        # Prepare a valid input so the simulated dependency failure is the only error
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)
        import_optional = vector_pd_accessor.import_optional

        def _missing_dask_geopandas(
            import_name: str, package_name: str | None = None, extra_name: str = "opt"
        ) -> object:
            """Fail only the Dask-GeoPandas import while forwarding other imports."""

            if import_name == "dask_geopandas":
                raise ImportError("Optional dependency 'dask-geopandas' required.")
            return import_optional(import_name, package_name=package_name, extra_name=extra_name)

        # Replace the module-level optional importer used by ``open_pointcloud``
        monkeypatch.setattr(vector_pd_accessor, "import_optional", _missing_dask_geopandas)
        with pytest.raises(ImportError, match="Optional dependency 'dask-geopandas' required.*"):
            gu.open_pointcloud(temp_file, data_column="z", chunks=5)

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_open_pointcloud_las__dask(self) -> None:
        """Open LAS data into lazy partitions while retaining file metadata."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Compare lazy metadata with the same source interpreted by PointCloud
        ds = gu.open_pointcloud(self.fn_las, chunks=100)
        pc = gu.PointCloud(self.fn_las)

        assert isinstance(ds, dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert ds.pc.point_count == pc.point_count
        # Computing all partitions should recover every source point
        assert len(ds.compute()) == pc.point_count
        assert not ds.pc.is_loaded

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_reproject_pointcloud_las__dask_geopandas(self) -> None:
        """Reproject a partitioned LAS point cloud and match the eager result."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Load an eager reference independently of the lazy LAS partitions
        ds = gu.open_pointcloud(self.fn_las, chunks=100)
        pc = gu.PointCloud(self.fn_las)
        pc.load()

        # Reprojection should stay lazy until equality requires computation
        reprojected = ds.pc.reproject(crs=3857)

        assert isinstance(reprojected, dgpd.GeoDataFrame)
        assert not reprojected.pc.is_loaded
        assert_geodataframe_equal(reprojected.compute(), pc.ds.to_crs(3857))
        assert not ds.pc.is_loaded
        assert not reprojected.pc.is_loaded

    @pytest.mark.skipif(find_spec("laspy") is None, reason="Only runs if laspy is installed.")
    def test_load_las__multiprocessing(self) -> None:
        """Load LAS row chunks through multiprocessing with the same eager result."""

        # Load the source once by worker chunks and once through the regular path
        pc_chunked = gu.PointCloud(self.fn_las)
        assert not pc_chunked.is_loaded
        pc_chunked.load(mp_config=MultiprocConfig(chunks=100))
        assert pc_chunked.is_loaded

        pc = gu.PointCloud(self.fn_las)
        pc.load()

        # Chunk scheduling must not change point order, values or metadata
        assert pc_chunked.pointcloud_equal(pc)
