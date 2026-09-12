"""Tests on the Pandas accessor mirroring the PointCloud API."""

from __future__ import annotations

import os.path
import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS

import geoutils as gu
import geoutils.vector.pd_accessor as vector_pd_accessor
from geoutils._misc import import_optional
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

    @pytest.mark.parametrize("suffix", [".las", ".laz"])
    @pytest.mark.parametrize("columns", ["main", "all", ["Z", "intensity"]])
    def test_open_pointcloud__empty_las_dask(
        self, tmp_path: Path, suffix: str, columns: Literal["main", "all"] | list[str]
    ) -> None:
        """Checks that empty LAS/LAZ files open lazily with the same columns, dtypes and CRS as eager reading."""

        laspy = pytest.importorskip("laspy")
        if suffix == ".laz":
            pytest.importorskip("lazrs")
        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # Write a valid LAS header without any point records
        # LasPy provides an independent fixture for the GeoUtils reader, including optional LAZ compression
        path = tmp_path / ("empty" + suffix)
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.add_crs(CRS.from_epsg(32633))
        laspy.LasData(header).write(path)
        expected = gu.open_pointcloud(str(path), columns=columns)

        # Build a lazy collection and inspect metadata without executing a partition
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            source = gu.open_pointcloud(str(path), columns=columns, chunks=3)
            assert isinstance(source, dgpd.GeoDataFrame)
            assert source.pc.point_count == 0
            assert source.pc.crs == expected.crs
        assert tasks == []
        graph = source.expr

        # Compute the empty collection and check its columns, types and unchanged lazy source
        assert_geodataframe_equal(source.compute(), expected)
        assert source.expr is graph and not source.pc.is_loaded

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

    def test_accessor__rejects_invalid_dask_dataframes(self) -> None:
        """Reject plain Dask DataFrames and validate lazy geometry partitions as points."""

        dgpd = pytest.importorskip("dask_geopandas")
        import dask.dataframe as dd

        vector_pd_accessor._register_dask_vector_accessor()
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        _register_dask_pointcloud_accessor()
        plain = dd.from_pandas(self.gdf.drop(columns="geometry"), npartitions=2)
        with pytest.raises(AttributeError, match="Dask-GeoPandas"):
            _ = plain.pc

        polygons = gpd.GeoDataFrame({"z": [1]}, geometry=[self.gdf.geometry.iloc[0].buffer(1)], crs=4326)
        lazy_polygons = dgpd.from_geopandas(polygons, npartitions=1)
        with pytest.raises(ValueError, match="point geometries"):
            lazy_polygons.pc.load()

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
        """Checks that lazy copies have the same location metadata while cropping and translation recalculate it."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from dask.callbacks import Callback

        # Open one on-disk source lazily and build the expected result through the eager accessor
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        self.gdf.to_file(temp_file)
        ds = gu.open_pointcloud(temp_file, data_column="z", chunks=5)
        expected = getattr(self.gdf.pc, method)(**kwargs)
        source_count, source_bounds = ds.pc.point_count, ds.pc.bounds

        # Each dataframe operation should add work without evaluating any point partition
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            output = getattr(ds.pc, method)(**kwargs)
            assert output.pc.crs == expected.crs
            assert output.pc.data_column == "z"
            if method == "copy":
                # Changing only values does not move points, so copies can reuse counts and bounds
                assert output.pc.bounds == source_bounds
                assert output.pc.point_count == source_count
                value_copy = ds.pc.copy(new_array=ds.pc.data * 2)
                assert value_copy.pc.bounds == source_bounds
                assert value_copy.pc.point_count == source_count
            else:
                assert output.pc.bounds is None
        assert tasks == []
        assert isinstance(output, dgpd.GeoDataFrame)
        assert not ds.pc.is_loaded
        assert not output.pc.is_loaded

        # Recount selected rows only on request; the original file's cached count and bounds are unchanged
        assert output.pc.point_count == len(expected)
        assert ds.pc.point_count == source_count
        assert ds.pc.bounds == source_bounds
        assert_geodataframe_equal(output.compute(), expected)
        assert not ds.pc.is_loaded
        assert not output.pc.is_loaded

        # Compute replacement values only on request and preserve every original point coordinate
        if method == "copy":
            expected_values = self.gdf.copy()
            expected_values["z"] *= 2
            assert_geodataframe_equal(value_copy.compute(), expected_values)

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


class TestPointCloudElevationMetadata:
    """
    Test module for elevation column and CRS metadata owned by the Pandas point cloud accessor.

    The tests cover direct Dask GeoDataFrames with empty and populated partitions, independent CRS metadata after
    reprojection, and explicit use of 3D geometry when auxiliary numeric columns are present. Dask checks also keep
    the original source graph and avoid computing partitions for metadata-only operations.
    """

    @pytest.mark.parametrize("point_count", [0, 1, 5])
    def test_crs__dask_geometry_metadata(self, point_count: int) -> None:
        """Checks that Dask point accessors read existing geometry CRS without a file metadata cache or computation."""

        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # Build a Dask GeoDataFrame directly, including empty and single-point collections
        coordinates = np.arange(point_count, dtype=float)
        frame = gu.PointCloudAccessor.from_xyz(
            500000 + coordinates * 20,
            8600000 + coordinates * 20,
            coordinates,
            crs="EPSG:32633+5703",
            data_column="height",
        )
        frame["intensity"] = coordinates + 100
        _register_dask_pointcloud_accessor()
        source = dgpd.from_geopandas(frame, chunksize=2)
        graph = source.expr

        # Read CRS and select elevations using metadata alone
        # File readers populate a GeoUtils CRS cache, but direct dataframe construction must also work
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            assert source.pc.crs == frame.crs
            source.pc.set_data_column("height")
            assert source.pc.crs == frame.crs

        # Check that metadata reads keep the original CRS and source graph without loading any partitions
        assert tasks == []
        assert source.expr is graph and not source.pc.is_loaded
        assert source.crs == frame.crs

    @pytest.mark.parametrize("lazy", [False, True])
    def test_reproject__metadata_is_independent(self, lazy: bool) -> None:
        """Checks that point cloud reprojection changes only the result CRS and keeps the source metadata."""

        # A small projected point cloud makes both the reference coordinates and metadata deterministic
        frame = gu.PointCloudAccessor.from_xyz([500000.0, 500020.0], [8600000.0, 8600020.0], [10.0, 20.0], crs=32633)
        source = frame
        if lazy:
            dgpd = pytest.importorskip("dask_geopandas")
            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor,
            )

            _register_dask_pointcloud_accessor()
            source = dgpd.from_geopandas(frame, npartitions=2)
        original_crs = source.pc.crs
        assert original_crs == frame.crs

        # Reprojection must not reuse a mutable CRS cache belonging to the source accessor
        result = source.pc.reproject(crs=32632)
        computed = result.compute() if lazy else result
        assert_geodataframe_equal(computed, frame.to_crs(32632))
        assert source.pc.crs == original_crs
        assert result.pc.crs == computed.crs
        if lazy:
            assert not source.pc.is_loaded and not result.pc.is_loaded

    def test_data_column__explicit_geometry_elevations(self) -> None:
        """Checks that 3D points can switch between geometry heights and a named data column."""

        # Keep elevations in 3D geometry and a distinct auxiliary column that must not become the main data
        frame = gu.PointCloudAccessor.from_xyz([1.0, 2.0], [3.0, 4.0], [10.0, 20.0], crs=32633, use_z=True)
        frame["intensity"] = np.array([2, 4], dtype=np.uint16)
        assert frame.pc.data_column is None
        np.testing.assert_array_equal(frame.pc.data, [10.0, 20.0])

        # Select intensity through another accessor; the shared dataframe keeps the new choice without changing Z
        other = gu.PointCloudAccessor(frame)
        other.set_data_column("intensity")
        assert frame.pc.data_column == "intensity"
        np.testing.assert_array_equal(frame.geometry.z, [10.0, 20.0])

        # Switch back to geometry height and check that a copied accessor sees the same active values
        other.set_data_column(None)
        assert frame.pc.data_column is None
        np.testing.assert_array_equal(frame.pc.copy().pc.data, [10.0, 20.0])
