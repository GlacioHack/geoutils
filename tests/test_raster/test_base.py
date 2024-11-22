"""Test RasterBase class, parent of Raster class and 'rst' Xarray accessor."""
from __future__ import annotations

import warnings
from typing import Any

import pytest
from pyproj import CRS
import numpy as np
import xarray as xr

from geoutils import Vector, Raster, open_raster
from geoutils import examples

class TestRasterBase:

    pass


def equal_xr_raster(ds: xr.DataArray, rast: Raster, warn_failure_reason: bool = True) -> bool:
    """Check equality of a Raster object and Xarray object"""

    # TODO: Move to raster_equal?
    equalities = [
        np.allclose(ds.data, rast.get_nanarray(), equal_nan=True),
        ds.rst.transform == rast.transform,
        ds.rst.crs == rast.crs,
        ds.rst.nodata == rast.nodata,
    ]

    names = ["data", "transform", "crs", "nodata"]

    complete_equality = all(equalities)

    if not complete_equality and warn_failure_reason:
        where_fail = np.nonzero(~np.array(equalities))[0]
        warnings.warn(
            category=UserWarning, message=f"Equality failed for: {', '.join([names[w] for w in where_fail])}."
        )
        print(f"Equality failed for: {', '.join([names[w] for w in where_fail])}.")

    print(np.count_nonzero(np.isfinite(ds.data) != np.isfinite(rast.get_nanarray())))
    print(np.nanmin(ds.data - rast.get_nanarray()))
    print(ds.data)

    return complete_equality

def output_equal(output1: Any, output2: Any) -> bool:
    """Return equality of different output types."""

    # For two vectors
    if isinstance(output1, Vector) and isinstance(output2, Vector):
        return output1.vector_equal(output2)

    # For two raster: Xarray or Raster objects
    elif isinstance(output1, Raster) and isinstance(output2, Raster):
        return output1.raster_equal(output2)
    elif isinstance(output1, Raster) and isinstance(output2, xr.DataArray):
        return equal_xr_raster(ds=output2, rast=output1)
    elif isinstance(output1, xr.DataArray) and isinstance(output2, Raster):
        return equal_xr_raster(ds=output1, rast=output2)

    # For arrays
    elif isinstance(output1, np.ndarray):
        return np.array_equal(output1, output2, equal_nan=True)

    # For tuple of arrays
    elif isinstance(output1, tuple) and isinstance(output1[0], np.ndarray):
        return np.array_equal(np.array(output1), np.array(output2), equal_nan=True)

    # For any other object type
    else:
        return output1 == output2

class TestClassVsAccessorConsistency:
    """
    Test class to check the consistency between the outputs of the Raster class and Xarray accessor for the same
    attributes or methods.

    All shared attributes should be the same.
    All operations manipulating the array should yield a comparable results, accounting for the fact that Raster class
    relies on masked-arrays and the Xarray accessor on NaN arrays.
    """

    # Run tests for different rasters
    landsat_b4_path = examples.get_path("everest_landsat_b4")
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")

    # Test common attributes
    attributes = ["crs", "transform", "nodata", "area_or_point", "res", "count", "height", "width", "footprint",
                  "shape", "bands", "indexes", "_is_xr", "is_loaded"]

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    @pytest.mark.parametrize("attr", attributes)  # type: ignore
    def test_attributes(self, path_raster: str, attr: str) -> None:
        """Test that attributes of the two objects are exactly the same."""

        # Open
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get attribute for each object
        output_raster = getattr(raster, attr)
        output_ds = getattr(getattr(ds, "rst"), attr)

        # Assert equality
        if attr != "_is_xr":  # Only attribute that is (purposely) not the same, but the opposite
            assert output_equal(output_raster, output_ds)
        else:
            assert output_raster != output_ds


    # Test common methods
    methods_and_args = {
        "reproject": {"crs": CRS.from_epsg(32610), "res": 10}}
    # methods_and_args = {
    #     "reproject": {"crs": CRS.from_epsg(32610), "res": 10},
    #     "crop": {"crop_geom": "random"},
    #     "translate": {"xoff": 10.5, "yoff": 5},
    #     "xy2ij": {"x": "random", "y": "random"},  # This will be derived during the test to work on all inputs
    #     "ij2xy": {"i": [0, 1, 2, 3], "j": [4, 5, 6, 7]},
    #     "coords": {"grid": True},
    #     "get_metric_crs": {"local_crs_type": "universal"},
    #     "reduce_points": {"points": "random"},  # This will be derived during the test to work on all inputs
    #     "interp_points": {"points": "random"},  # This will be derived during the test to work on all inputs
    #     "proximity": {"target_values": [100]},
    #     "outside_image": {"xi": [-2, 10000, 10], "yj": [10, 50, 20]},
    #     "to_pointcloud": {"subsample": 1000, "random_state": 42},
    #     "polygonize": {"target_values": "all"},
    #     "subsample": {"subsample": 1000, "random_state": 42},
    # }

    @pytest.mark.parametrize("path_raster", [aster_dem_path])  # type: ignore
    @pytest.mark.parametrize("method", list(methods_and_args.keys()))  # type: ignore
    def test_methods(self, path_raster: str, method: str) -> None:
        """
        Test that the outputs of the two objects are exactly the same
        (converted for the case of a raster/vector output, as it can be a Xarray/GeoPandas object or Raster/Vector).
        """

        # Open both objects
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

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

        elif "crop_geom" in self.methods_and_args[method].keys():
            crop_geom = raster.bounds.left + 100, raster.bounds.bottom + 200, \
                raster.bounds.left + 320, raster.bounds.bottom + 411
            args = self.methods_and_args[method].copy()
            args.update({"crop_geom": crop_geom})

        else:
            args = self.methods_and_args[method].copy()

        # Apply method for each class
        output_raster = getattr(raster, method)(**args)
        output_ds = getattr(getattr(ds, "rst"), method)(**args)

        # Assert equality of output
        assert output_equal(output_raster, output_ds)

