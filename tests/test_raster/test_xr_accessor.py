"""Tests on Xarray accessor mirroring Raster API."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils import examples, open_raster


class TestAccessor:
    """
    Test for Xarray accessor subclass.

    Note: This test class only tests functionalities that are specific to the RasterAccessor subclass. Overridden
    abstract methods, loading behaviour and Dask laziness are tested in test_base directly to mirror Raster tests.

    This class thus tests:
    - The open_raster function,
    - The instantiation __init__ through ds.rst,
    - The to_geoutils() method.
    """

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_open_raster(self) -> None:
        pass

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_copy(self, path_raster: str) -> None:

        ds = open_raster(path_raster)
        ds_copy = ds.rst.copy()

        assert np.array_equal(ds.data, ds_copy.data, equal_nan=True)
        assert ds.rst.transform == ds_copy.rst.transform
        assert ds.rst.crs == ds_copy.rst.crs
        assert ds.rst.nodata == ds_copy.rst.nodata

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_open__loaded(self, path_raster: str) -> None:
        """
        Test that a DataArray opened using "open_raster" maintains implicit loading logic.

        Tests checking loading for all attributes and methods are done in TestBase.

        Note: this is different from using lazy Dask arrays: for any array type, Xarray only loads metadata, and
        implicitly loads data in memory when .data or .load() is called.
        """

        # Open raster with/without chunks, should not load in memory yet
        ds = open_raster(path_raster)
        assert not ds._in_memory

        # The array should be NumPy
        assert isinstance(ds.data, np.ndarray)
        ds.load()
        assert ds._in_memory

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_open__dask(self, path_raster: str) -> None:
        """
        Check that a DataArray opened with chunks using "open_raster" maintains Dask laziness.

        Note: this is different from loading mechanism of Xarray (triggers when calling .data).
        """
        pytest.importorskip("dask")
        import dask.array as da

        # Open raster lazily with chunks
        ds = open_raster(path_raster, chunks={"band": 1, "x": 10, "y": 10})

        # Array should be a Dask array (chunks exist)
        ds_arr = ds.data
        assert not ds._in_memory
        assert isinstance(ds_arr, da.Array)
        assert ds_arr.chunks is not None

        # After compute, it should be a NumPy array
        ds_comp = ds.compute()
        assert isinstance(ds_comp.data, np.ndarray)
        assert ds_comp._in_memory

    def test_open__dask_nodata_can_be_written(self, tmp_path: Path) -> None:
        """Write a lazily opened nodata raster without conflicting xarray metadata."""

        pytest.importorskip("dask")

        # Opening a masked file moves its encoded nodata to one unambiguous attribute
        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})
        assert ds.rst.nodata is not None
        assert "_FillValue" not in ds.encoding

        # The final writer must preserve nodata while computing the Dask array
        output_file = tmp_path / "dask-nodata.tif"
        ds.rst.to_file(output_file)
        with rio.open(output_file) as output:
            assert output.nodata == ds.rst.nodata

    def test_cross_type_outputs_are_accessors(self) -> None:
        ds = gu.RasterAccessor.from_array(
            data=np.array([[1, 1], [0, 0]], dtype=np.uint8),
            transform=from_origin(0, 2, 1, 1),
            crs=4326,
            nodata=None,
        )

        vector = ds.rst.polygonize(target_values=1)
        assert isinstance(vector, gpd.GeoDataFrame)
        assert vector.vct.to_geoutils().vector_equal(gu.Vector(vector))

        pointcloud = ds.rst.to_pointcloud(skip_nodata=False)
        assert isinstance(pointcloud, gpd.GeoDataFrame)
        assert pointcloud.pc.data_column == "b1"

        interpolated = ds.rst.interp_points((np.array([0.5]), np.array([1.5])), method="nearest")
        assert isinstance(interpolated, gpd.GeoDataFrame)
        assert interpolated.pc.data_column == "z"

        footprint = ds.rst.get_footprint_projected(ds.rst.crs)
        assert isinstance(footprint, gpd.GeoDataFrame)

    def test_get_stats__dask_global_quantile_stats(self) -> None:
        """Regression test for Dask-backed xarray stats that require global quantiles."""

        pytest.importorskip("dask")
        import dask.array as da

        base = open_raster(self.aster_dem_path)
        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})

        assert isinstance(ds.data, da.Array)
        assert not ds._in_memory

        for stat in ["median", "90th percentile", "le90", "nmad", "iqr"]:
            expected = float(base.rst.get_stats(stat))
            actual = ds.rst.get_stats(stat)

            assert isinstance(actual, da.Array)
            assert float(actual.compute()) == pytest.approx(expected, rel=1e-5)
            assert not ds._in_memory

    def test_reproject__dask_keeps_dimension_order_for_stats(self) -> None:
        """Regression test for Dask xarray reprojection outputs with valid rioxarray dimension order."""

        pytest.importorskip("dask")

        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})

        reprojected_crs = ds.rst.reproject(crs=4326)
        reprojected_res = ds.rst.reproject(res=(ds.rst.res[0] * 2, ds.rst.res[1] / 2), resampling="bilinear")

        for reprojected in [reprojected_crs, reprojected_res]:
            assert reprojected.dims == ("y", "x")
            assert hasattr(reprojected.rst.get_stats("mean"), "compute")
            assert np.isfinite(float(reprojected.rst.get_stats("mean").compute()))

    def test_chunked_rasterize_paths_accept_dask_chunk_tuples(self) -> None:
        """Regression test for xarray/Dask rasterization paths receiving normalized chunk tuples."""

        pytest.importorskip("dask")
        import dask.array as da

        arr = np.zeros((12, 10), dtype=np.uint8)
        arr[2:9, 3:8] = 1
        dask_arr = da.from_array(arr, chunks=(5, 4))
        ds = gu.RasterAccessor.from_array(
            data=dask_arr,
            transform=from_origin(0, 12, 1, 1),
            crs=4326,
            nodata=0,
        )
        vector = gu.Vector(gpd.GeoDataFrame({"geometry": [box(2, 4, 9, 11)]}, crs=4326))

        mask = vector.create_mask(ds.rst, dask=True)
        assert isinstance(mask.data, da.Array)
        assert mask.data.chunks == dask_arr.chunks
        assert bool(mask.compute().data[3, 3])

        polygons = ds.rst.polygonize(target_values=1)
        rasterized = polygons.vct.rasterize(ds.rst, in_value=1, out_value=0, out_dtype=np.uint8, dask=True)
        assert isinstance(rasterized.data, da.Array)
        assert rasterized.data.chunks == dask_arr.chunks
        assert np.array_equal(rasterized.compute().data, arr)
