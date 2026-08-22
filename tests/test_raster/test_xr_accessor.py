"""Tests on Xarray accessor mirroring Raster API."""

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin

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
