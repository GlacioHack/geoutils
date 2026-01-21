"""Test RasterBase class, parent of Raster class and 'rst' Xarray accessor."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal
from pyproj import CRS

from geoutils import Raster, Vector, examples, open_raster
from geoutils.raster.georeferencing import _default_nodata


def equal_xr_raster(ds: xr.DataArray, rast: Raster, warn_failure_reason: bool = True) -> bool:
    """Check equality of a Raster object and Xarray object"""

    # TODO: Move to raster_equal?
    equalities = [
        np.allclose(ds.data, rast.get_nanarray(), equal_nan=True),
        ds.rst.transform == rast.transform,
        ds.rst.crs == rast.crs,
        ds.rst.nodata == rast.nodata,
        np.array_equal(~np.isfinite(ds.data), np.ma.getmaskarray(rast.data)),
    ]

    names = ["data", "transform", "crs", "nodata", "mask"]

    complete_equality = all(equalities)

    if not complete_equality and warn_failure_reason:
        where_fail = np.nonzero(~np.array(equalities))[0]
        warnings.warn(
            category=UserWarning, message=f"Equality failed for: {', '.join([names[w] for w in where_fail])}."
        )
        print(f"Equality failed for: {', '.join([names[w] for w in where_fail])}.")
        if not equalities[0]:
            diff = ds.data - rast.get_nanarray()
            valids = np.isfinite(diff)
            print(f"Number of non-equal pixels: {np.sum(diff[valids] != 0)}")
            print(f"Mean: {np.nanmean(diff[valids])}")
            print(f"Absolute percentile 90: {np.nanpercentile(np.abs(diff[valids]), 90)}")

    return complete_equality


def assert_output_equal(output1: Any, output2: Any) -> None:
    """Return equality of different output types."""

    # For two vectors
    if isinstance(output1, Vector) and isinstance(output2, Vector):
        assert output1.vector_equal(output2)

    # For two raster: Xarray or Raster objects
    elif isinstance(output1, Raster) and isinstance(output2, Raster):
        assert output1.raster_equal(output2)
    elif isinstance(output1, Raster) and isinstance(output2, xr.DataArray):
        assert equal_xr_raster(ds=output2, rast=output1)
    elif isinstance(output1, xr.DataArray) and isinstance(output2, Raster):
        assert equal_xr_raster(ds=output1, rast=output2)

    # For arrays
    elif isinstance(output1, np.ndarray):
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


class TestClassVsAccessorConsistency:
    """
    Test class to check the consistency between the outputs of the Raster class and Xarray accessor for the same
    attributes or methods.

    All shared attributes should be the same.
    All operations manipulating the array should yield a comparable results, accounting for the fact that Raster class
    relies on masked-arrays and the Xarray accessor on NaN arrays.
    """

    # Run tests for different rasters
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")

    # Test common attributes
    attributes = [
        "crs",
        "transform",
        "nodata",
        "area_or_point",
        "res",
        "count",
        "height",
        "width",
        "footprint",
        "shape",
        "bands",
        "indexes",
        "_is_xr",
    ]

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("attr", attributes)  # type: ignore
    def test_attributes_consistency(self, path_raster: str, attr: str) -> None:
        """Test that attributes of the two objects are exactly the same between a Raster and Xarray rst accessor."""

        # Open
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get attribute for each object
        output_raster = getattr(raster, attr)
        output_ds = getattr(ds.rst, attr)

        # Assert equality
        if attr == "_is_xr":  # Only attribute that is (purposely) not the same, but the boolean opposite
            assert output_raster != output_ds
        else:
            assert_output_equal(output_raster, output_ds)

    # Test common methods
    methods_and_args = {
        "reproject": {"crs": CRS.from_epsg(32610), "res": 10},
        "crop": {"bbox": "random"},  # This will be derived during the test to work on all inputs
        "icrop": {"bbox": "random"},  # This will be derived during the test to work on all inputs
        "translate": {"xoff": 10.5, "yoff": 5},
        "xy2ij": {"x": "random", "y": "random"},  # This will be derived during the test to work on all inputs
        "ij2xy": {"i": [0, 1, 2, 3], "j": [4, 5, 6, 7]},
        "coords": {"grid": True},
        "get_metric_crs": {"local_crs_type": "universal"},
        "get_nanarray": {},
        # "reduce_points": {"points": "random"},  # This will be derived during the test to work on all inputs
        "interp_points": {"points": "random"},  # This will be derived during the test to work on all inputs
        "proximity": {"target_values": [100]},
        "outside_image": {"xi": [-2, 10000, 10], "yj": [10, 50, 20]},
        "to_pointcloud": {"subsample": 1000, "random_state": 42},
        "polygonize": {"target_values": "all"},
        "subsample": {"subsample": 1000, "random_state": 42},
        "filter": {"method": "median", "size": 7},
        "get_stats": {},
    }

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("method", list(methods_and_args.keys()))  # type: ignore
    def test_methods_consistency(self, path_raster: str, method: str) -> None:
        """
        Test that the method output of the two objects are exactly the same between a Raster and Xarray rst accessor
        (converted for the case of a raster/vector output, as it can be a Xarray/GeoPandas object or Raster/Vector).
        """

        # Open both objects
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # If integer type in Raster, convert to float32 to match Xarray accessor behaviour
        if "int" in str(raster.dtype):
            raster = raster.astype(dtype=np.float32, convert_nodata=False)

        # If nodata is not defined, define one
        if raster.nodata is None:
            ds.rst.set_nodata(_default_nodata(ds.rst.dtype))
            raster.set_nodata(_default_nodata(ds.rst.dtype))

        # Loop for specific inputs that require knowledge of the data
        if "points" in self.methods_and_args[method].keys() or "x" in self.methods_and_args[method].keys():
            rng = np.random.default_rng(seed=42)
            ninterp = 10
            res = raster.res
            interp_x = (rng.choice(raster.shape[0], ninterp) + rng.random(ninterp)) * res[0]
            interp_y = (rng.choice(raster.shape[1], ninterp) + rng.random(ninterp)) * res[1]
            args = self.methods_and_args[method].copy()
            if "points" in self.methods_and_args[method].keys():
                args.update({"points": (interp_x, interp_y)})
            elif "x" in self.methods_and_args[method].keys():
                args.update({"x": interp_x, "y": interp_y})

        elif method == "crop":
            bbox = (
                raster.bounds.left + 100,
                raster.bounds.bottom + 200,
                raster.bounds.left + 320,
                raster.bounds.bottom + 411,
            )
            args = self.methods_and_args[method].copy()
            args.update({"bbox": bbox})

        elif method == "icrop":
            bbox = 3, 5, 10, 22
            args = self.methods_and_args[method].copy()
            args.update({"bbox": bbox})

        else:
            args = self.methods_and_args[method].copy()

        # Apply method for each class
        output_raster = getattr(raster, method)(**args)
        output_ds = getattr(ds.rst, method)(**args)

        # Assert equality of output
        assert_output_equal(output_raster, output_ds)
