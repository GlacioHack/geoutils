"""Tests on Xarray accessor mirroring Raster API."""

import warnings

import dask.array as da
import numpy as np
import pytest

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

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_copy(self, path_raster: str) -> None:

        ds = open_raster(path_raster)
        ds_copy = ds.rst.copy()

        assert np.array_equal(ds.data, ds_copy.data, equal_nan=True)
        assert ds.rst.transform == ds_copy.rst.transform
        assert ds.rst.crs == ds_copy.rst.crs
        assert ds.rst.nodata == ds_copy.rst.nodata

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])  # type: ignore
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

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_open__dask(self, path_raster: str) -> None:
        """
        Check that a DataArray opened with chunks using "open_raster" maintains Dask laziness.

        Note: this is different from loading mechanism of Xarray (triggers when calling .data).
        """

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

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_reproject__dask(self, path_raster: str) -> None:
        """
        Check that reproject maintains Dask laziness.
        """

        warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata.*")

        # Open raster lazily with chunks
        ds = open_raster(path_raster, chunks={"band": 1, "x": 10, "y": 10})

        ds_reproj = ds.rst.reproject(res=50)

        # This shouldn't affect the input dataset, that remains unloaded
        ds_arr = ds.data
        assert not ds._in_memory
        assert isinstance(ds_arr, da.Array)

        # And the created output should also be lazy
        ds_arr_reproj = ds_reproj.data
        assert isinstance(ds_arr_reproj, da.Array)
        assert ds_arr_reproj.chunks is not None

        # And computes successfully
        ds_reproj_comp = ds_reproj.compute()
        assert isinstance(ds_reproj_comp.data, np.ndarray)
        assert ds_reproj_comp._in_memory
