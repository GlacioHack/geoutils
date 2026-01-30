"""Test RasterBase class, parent of Raster class and 'rst' Xarray accessor."""

from __future__ import annotations

import os
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
import xarray as xr
from packaging.version import Version
from pandas.testing import assert_frame_equal
from pyproj import CRS

from geoutils import Raster, Vector, examples, open_raster
from geoutils.raster import MultiprocConfig
from geoutils.raster.base import RasterBase
from geoutils.raster.xr_accessor import RasterAccessor


@pytest.fixture(scope="module")
def lazy_test_files(tmp_path_factory: Any) -> list[str]:
    """
    Create temporary converted files for lazy tests.

    Below, we compare Xarray accessor and Raster class including loading/laziness (without data loaded).
    So we need to convert all of our integer test examples to float32 with valid nodata ahead of loading,
    and save them to temporary test files.
    """

    # Create temporary directory at module scope
    tmpdir = tmp_path_factory.mktemp("lazy_data")

    list_name = ["everest_landsat_b4", "everest_landsat_rgb", "exploradores_aster_dem"]
    list_fn_out = []
    for name in list_name:

        # Get filepath
        fn = examples.get_path_test(name)

        # If dataset is already float32 with defined nodata, return the path
        if name == "exploradores_aster_dem":
            list_fn_out.append(fn)
            continue

        # Else open, convert
        rast = Raster(fn)
        rast = rast.astype(dtype=np.float32, convert_nodata=False)
        rast.set_nodata(-9999)

        # Save to file in temporary directory
        fn_out = os.path.join(tmpdir, os.path.splitext(os.path.basename(fn))[0] + "_float32.tif")
        rast.to_file(fn_out)

        list_fn_out.append(fn_out)

    return list_fn_out


def assert_output_equal(output1: Any, output2: Any, use_allclose: bool = False, strict_masked: bool = True) -> None:
    """Return equality of different output types."""

    # For two vectors
    if isinstance(output1, Vector) and isinstance(output2, Vector):
        assert output1.vector_equal(output2)

    # For two raster: Xarray or Raster objects
    elif isinstance(output1, (Raster, xr.DataArray)):
        if use_allclose:
            assert output1.raster_allclose(output2, warn_failure_reason=True, strict_masked=strict_masked)
        else:
            assert output1.raster_equal(output2, warn_failure_reason=True, strict_masked=strict_masked)

    # For arrays
    elif isinstance(output1, np.ndarray):
        if np.ma.isMaskedArray(output1):
            output1 = output1.filled(np.nan)
        if np.ma.isMaskedArray(output2):
            output2 = output2.filled(np.nan)
        assert np.array_equal(output1, output2, equal_nan=True)

    # For tuple of arrays
    elif isinstance(output1, tuple) and isinstance(output1[0], np.ndarray):
        assert np.array_equal(np.array(output1), np.array(output2), equal_nan=True)

    # For a dictionary of numeric values
    elif isinstance(output1, dict):
        df1 = pd.DataFrame(index=[0], data=output1)
        df2 = pd.DataFrame(index=[0], data=output2)
        assert_frame_equal(df1, df2, check_dtype=False)
    # For any other object type
    else:
        assert output1 == output2


def should_be_loaded(method: str, args: dict[str, Any], noload: list[str], noload_allowed_args: dict[str, Any]) -> bool:
    """Helper function to check without a method input/output should be loaded or not, based on input dictionaries."""

    # For method where the behaviour is independent of their arguments
    if method not in noload_allowed_args.keys():
        # If the method has a single behaviour, simply check if it belongs in the list
        should_output_be_loaded = method not in noload
    # For method where the behaviour depends on their arguments
    else:
        # Get relevant method arguments
        allowed = noload_allowed_args[method]
        # If any value is different from the list of values in the allowed dictionary, it should load
        any_different = not all(
            (not isinstance(args[k], np.ndarray) and args[k] in allowed[k]) for k in allowed if k in args
        )
        should_output_be_loaded = any_different

    return should_output_be_loaded


class NeedsTestError(ValueError):
    """Error to remember to add test when a new RasterBase method is added."""


class TestClassVsAccessorConsistency:
    """
    Test class to check the consistency between the outputs, loading, laziness and chunked operations
    of the Raster class and Xarray accessor for the same attributes or methods.

    All shared attributes should be the same.
    All operations manipulating the array should yield a comparable results, accounting for the fact that Raster class
    relies on masked-arrays and the Xarray accessor on NaN arrays (converted to float32 if integer type).
    """

    # Get all RasterBase public properties and methods, ensures we test absolutely everything even with API changes
    # The full list of properties is used directly to test all methods
    properties = [k for k, v in RasterBase.__dict__.items() if not k.startswith("_") and isinstance(v, property)]
    # The full list of methods is used a posteriori to check all were tested across multiple tests
    methods = [k for k, v in RasterBase.__dict__.items() if not k.startswith("_") and not isinstance(v, property)]
    # Ignore deprecated methods (already tested through their new name)
    methods = [m for m in methods if m not in ["to_points", "save"]]

    # List of properties that WILL load the input dataset (only one does, the data itself)
    properties_input_load = ["data"]

    # List of methods that WILL NOT load the input dataset
    methods_input_noload = [
        "crop",
        "icrop",
        "translate",
        "get_metric_crs",
        "xy2ij",
        "ij2xy",
        "coords",
        "outside_image",
        "info",
        "get_bounds_projected",
        "get_footprint_projected",
        "copy",
        "georeferenced_grid_equal",
        "intersection",
    ]
    # List of methods that WILL NOT load the input for certain arguments
    methods_input_noload_allowed_args = {"info": {"stats": [False]}}

    # List of methods that WILL NOT load the output dataset (mostly in-place methods; otherwise rare,
    # only icrop/crop/translate, that correspond to isel/sel and assign_coords of Xarray)
    methods_output_noload = [
        "crop",
        "icrop",
        "translate",
        "copy",
        "set_crs",
        "set_transform",
        "set_nodata",
        "set_area_or_point",
    ]
    # List of methods that WILL NOT LOAD the output for certain arguments
    # copy(new_array=not None) will load
    methods_output_noload_allowed_args = {"copy": {"deep": [True, False], "new_array": [None]}}

    @pytest.mark.parametrize("path_index", [0, 1, 2])
    @pytest.mark.parametrize("prop", properties)
    def test_properties__equality_and_loading(self, path_index: int, prop: str, lazy_test_files: list[str]) -> None:
        """
        Test that properties are exactly equal between a Raster and DataArray using the "rst" accessor,
        and if they do not load the dataset or not.
        """

        # Get file path
        path_raster = lazy_test_files[path_index]

        # Open
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # Get attribute for each object
        output_raster = getattr(raster, prop)
        output_ds = getattr(ds.rst, prop)

        # Assert equality
        if prop == "_is_xr":  # Only attribute that is (purposely) not the same, but the boolean opposite
            assert output_raster != output_ds
        else:
            # Tags are not exactly the same (scale/offset added in Xarray; mostly "AREA_OR_POINT" in common)
            if prop in ["tags"]:
                assert all(output_ds[k] == v for k, v in output_raster.items() if k in ["AREA_OR_POINT"])
            # All others are exactly the same
            else:
                assert_output_equal(output_raster, output_ds)

        # Check getting attribute did not (or did) load the Raster or Xarray dataset
        should_be_loaded = prop in self.properties_input_load
        assert raster.is_loaded is should_be_loaded
        assert ds._in_memory is should_be_loaded

    # Test all methods that are not class methods
    methods_and_kwargs = [
        # 1/ This first list of methods do not load the input raster (no access of .data)
        # 1.1. Not in-place
        ("copy", {"deep": False}),  # Shallow copy does not load
        ("copy", {"deep": True}),  # Deep copy does not load either if unloaded!
        ("info", {"stats": False, "verbose": False}),  # Verbose false to capture output
        (
            "crop",
            {"bbox": "random"},
        ),  # "random" will be derived during the test to work on all inputs
        ("icrop", {"bbox": (3, 5, 10, 22)}),
        ("translate", {"xoff": 10.5, "yoff": 5}),
        ("xy2ij", {"x": "random", "y": "random"}),  # "random" will be derived during the test to work on all inputs
        ("ij2xy", {"i": [0, 1, 2, 3], "j": [4, 5, 6, 7]}),
        ("coords", {"grid": True}),
        ("get_metric_crs", {"local_crs_type": "universal"}),
        ("get_footprint_projected", {"out_crs": CRS.from_epsg(4326)}),
        ("get_bounds_projected", {"out_crs": CRS.from_epsg(4326)}),
        ("georeferenced_grid_equal", {"other": "self"}),
        ("outside_image", {"xi": [-2, 10000, 10], "yj": [10, 50, 20]}),
        # 1.2. In-place methods
        ("translate", {"xoff": 10.5, "yoff": 5, "inplace": True}),
        ("set_transform", {"new_transform": rio.transform.from_bounds(0, 0, 1, 1, 5, 5)}),
        ("set_crs", {"new_crs": CRS.from_epsg(4326)}),
        ("set_nodata", {"new_nodata": -10001, "update_array": False, "update_mask": False}),
        ("set_area_or_point", {"new_area_or_point": "Point"}),
        # 2/ This second list of methods will load the input Raster (access .data)
        # 2.1. Not in-place
        ("copy", {"new_array": "placeholder"}),  # Copy with new array does load! Will create array of right size below.
        ("info", {"stats": True, "verbose": False}),  # Info with stats loads
        ("reproject", {"crs": CRS.from_epsg(4326)}),
        ("raster_equal", {"other": "self"}),
        ("raster_allclose", {"other": "self"}),
        ("intersection", {"other": "self"}),
        # ("reduce_points", {"points": "random"}),  # Needs implementation in RasterBase (currently only for Raster)
        ("interp_points", {"points": "random"}),  # "random" will be derived during the test to work on all inputs
        ("proximity", {"target_values": [100]}),
        ("get_nanarray", {}),
        ("to_pointcloud", {"subsample": 1, "random_state": 42}),
        ("polygonize", {"target_values": "all"}),
        ("subsample", {"subsample": 1000, "random_state": 42}),
        ("filter", {"method": "median", "size": 7}),
        ("get_stats", {}),
        # 2.2. In-place methods
        ("load", {}),
    ]

    @pytest.mark.parametrize("path_index", [0, 1, 2])
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in methods_and_kwargs])
    def test_methods__equality_and_loading(
        self, path_index: int, method: str, kwargs: dict[str, Any], lazy_test_files: list[str]
    ) -> None:
        """
        Test that the method output and loading mechanism of the two objects are exactly the same between a Raster and
        Xarray rst accessor.

        For this test, the integer test file where converted ahead (to preserve loading mechanism and laziness) in the
        fixture "lazy_test_files".
        """

        # Get filepath
        path_raster = lazy_test_files[path_index]

        # Open both objects
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # For methods that require knowledge of the data (relative to bounds), create specific inputs
        args = kwargs.copy()
        if "points" in method or "xy2ij" in method:
            rng = np.random.default_rng(seed=42)
            ninterp = 10
            res = raster.res
            interp_x = raster.bounds.left + (rng.choice(raster.shape[0], ninterp) + rng.random(ninterp)) * res[0]
            interp_y = raster.bounds.bottom + (rng.choice(raster.shape[1], ninterp) + rng.random(ninterp)) * res[1]
            if "points" in method:
                args.update({"points": (interp_x, interp_y)})
            elif "xy2ij" in method:
                args.update({"x": interp_x, "y": interp_y})
        elif method == "crop":
            bbox = (
                raster.bounds.left + 100,
                raster.bounds.bottom + 200,
                raster.bounds.left + 320,
                raster.bounds.bottom + 411,
            )
            args.update({"bbox": bbox})
        elif method in ["raster_equal", "raster_allclose", "georeferenced_grid_equal", "intersection"]:
            args.update({"other": ds.copy(deep=False)})
        elif method == "copy" and "new_array" in args:
            args.update({"new_array": np.ones(ds.shape)})

        # Apply method for each class
        output_raster = getattr(raster, method)(**args)
        output_ds = getattr(ds.rst, method)(**args)

        # Determine if operation was in-place or not
        inplace = "inplace" in args or method in ["set_transform", "set_crs", "set_nodata", "set_area_or_point", "load"]
        # If yes, outputs should be None, and we'll check loading behaviour for inputs as if they were outputs
        if inplace:
            assert output_raster is None
            assert output_ds is None
            output_ds = ds
            output_raster = raster
        # If no, we check input status
        else:
            # Check using method did or did not load the input Raster or Xarray dataset, following expected values
            should_input_be_loaded = should_be_loaded(
                method=method,
                args=args,
                noload=self.methods_input_noload,
                noload_allowed_args=self.methods_input_noload_allowed_args,
            )
            assert raster.is_loaded is should_input_be_loaded
            assert ds._in_memory is should_input_be_loaded

        # In the case of a Raster / DataArray output, check if output is loaded or not
        # (apart from in-place metadata setting, only a few functions don't load output, such as: crop/icrop, copy,
        # translate; matching sel/isel, copy, assign_coords in Xarray)
        if isinstance(output_ds, xr.DataArray):
            should_output_be_loaded = should_be_loaded(
                method=method,
                args=args,
                noload=self.methods_output_noload,
                noload_allowed_args=self.methods_output_noload_allowed_args,
            )
            # TODO: Raster class does not load input, but does load output for "crop/icrop"
            if method not in ["crop", "icrop"]:
                assert output_raster.is_loaded is should_output_be_loaded
            assert output_ds._in_memory is should_output_be_loaded

        # Finally, assert exact equality of outputs
        # (in case of raster; this will load all the data, so has to come at the end)
        assert_output_equal(output_raster, output_ds)

    class_methods_and_kwargs = [
        (
            "from_array",
            {
                "data": np.ones((5, 5)),
                "transform": rio.transform.from_bounds(0, 0, 1, 1, 5, 5),
                "crs": CRS.from_epsg(4326),
                "nodata": -9999,
                "tags": {"metadata": "test"},
                "area_or_point": "Point",
            },
        ),
        (
            "from_pointcloud_regular",
            {
                "pointcloud": gpd.GeoDataFrame(
                    data={"b1": np.ones(4)},
                    geometry=gpd.points_from_xy(
                        x=[
                            0,
                            2,
                            0,
                            1,
                        ],
                        y=[3, 3, 4, 4],
                        crs=4326,
                    ),
                ),
                "grid_coords": (np.array([0, 1, 2]), np.array([3, 4])),
            },
        ),
    ]

    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in class_methods_and_kwargs])
    def test_classmethods__equality(self, method: str, kwargs: dict[str, Any]) -> None:
        """Test class methods output exactly the same objects. Loading always happens for class methods."""

        # Accessor only uses this internally, but we expose it as a class method anyway
        output_raster = getattr(Raster, method)(**kwargs)
        output_ds = getattr(RasterAccessor, method)(**kwargs)

        assert_output_equal(output_raster, output_ds)

    def test_methods__test_coverage(self) -> None:
        """Test that checks that all existing RasterBase methods are tested above."""

        # Compare tested methods from above list of tuples to all methods derived from class dictionary
        methods_1 = [m[0] for m in self.methods_and_kwargs]
        methods_2 = [m[0] for m in self.class_methods_and_kwargs]
        list_missing = [method for method in self.methods if method not in methods_1 + methods_2]

        if len(list_missing) != 0:
            raise AssertionError(f"RasterBase not covered by tests: {list_missing}")

    chunked_methods_and_args = (("reproject", {"crs": CRS.from_epsg(4326)}),)

    @pytest.mark.parametrize("path_index", [0, 2])
    @pytest.mark.parametrize("method, kwargs", [(f, k) for f, k in chunked_methods_and_args])
    def test_chunked_methods__equality_loading_laziness(
        self, path_index: int, method: str, kwargs: dict[str, Any], lazy_test_files: list[str]
    ) -> None:
        """
        Test that chunked methods have the exact same output, loading mechanism and laziness.

        They should yield the exact same output for:
        - In-memory,
        - Dask backend through Xarray accessor,
        - Multiprocessing backend through Raster class.

        Dask array should remain delayed before compute, and Multiprocessing output remains unloaded.
        """

        pytest.importorskip("dask")
        import dask.array as da

        # Get filepath
        path_raster = lazy_test_files[path_index]

        # Open lazily with Dask
        ds = open_raster(path_raster, chunks={"band": 1, "x": 10, "y": 10})
        # Open raster that will be processed using Multiprocessing
        mp_config = MultiprocConfig(chunk_size=10)  # To pass to the function
        raster = Raster(path_raster)
        # Open and load both DataArray/Raster with NumPy
        ds2 = open_raster(path_raster)
        ds2.load()
        raster2 = Raster(path_raster)
        raster2.load()

        # Apply method for each
        output_raster = getattr(raster, method)(**kwargs, multiproc_config=mp_config)
        output_ds = getattr(ds.rst, method)(**kwargs)
        output_raster2 = getattr(raster2, method)(**kwargs)
        output_ds2 = getattr(ds2.rst, method)(**kwargs)

        # 1/ For Dask object: both inputs and outputs should be unloaded + lazy, and compute
        # Input
        assert not ds._in_memory
        assert isinstance(ds.data, da.Array)
        assert ds.data.chunks is not None
        # Output
        assert not output_ds._in_memory
        assert isinstance(output_ds.data, da.Array)
        assert output_ds.data.chunks is not None
        # Output computes successfully, and is then loaded in memory
        output_ds = output_ds.compute()
        assert isinstance(output_ds.data, np.ndarray)
        assert output_ds._in_memory

        # 2/ For Multiprocessing, same for loading
        assert not raster.is_loaded
        assert not output_raster.is_loaded

        # 3/ For non-Dask array, both should be loaded
        assert ds2._in_memory
        assert isinstance(ds2.data, np.ndarray)
        assert output_ds2._in_memory
        assert isinstance(output_ds2.data, np.ndarray)

        # 4/ For raster, same
        assert raster2.is_loaded
        assert output_raster2.is_loaded

        # Check all outputs are exactly the same
        # (For reproject, no artefacts only since we added "tolerance" argument in Rasterio,
        # which officially came out in 1.5; so we skip the test for earlier versions)
        if method == "reproject" and Version(rio.__version__) < Version("1.5.0"):
            return
        assert_output_equal(output_raster, output_ds)  # Same chunk sizes, so exact same numerics
        assert_output_equal(output_raster, output_raster2, use_allclose=True, strict_masked=False)
        assert_output_equal(output_raster, output_ds2, use_allclose=True)

    match_methods = ["reproject", "crop", "create_mask", "rasterize", "proximity", "grid"]

    def test_methods__match_raster(self) -> None:
        """Test that methods that take a raster as match-reference input behave the same with DataArray/Raster:
        - Equal output,
        - No loading of the raster.
        """

        # TODO: Finalize after consistent input check function #850
        assert True
