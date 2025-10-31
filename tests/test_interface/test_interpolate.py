from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pytest
import rasterio as rio
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation

import geoutils as gu
from geoutils import examples
from geoutils.interface.interpolate import (
    _get_dist_nodata_spread,
    _interp_points,
    _interpn_interpolator,
    method_to_order,
)
from geoutils.projtools import reproject_to_latlon


class TestInterpolate:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_b4_crop_path = examples.get_path_test("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")

    def test_dist_nodata_spread(self) -> None:
        """Test distance of nodata spreading computation based on interpolation order."""

        assert _get_dist_nodata_spread(0, "half_order_up") == 0
        assert _get_dist_nodata_spread(0, "half_order_down") == 0
        assert _get_dist_nodata_spread(0, 5) == 5

        assert _get_dist_nodata_spread(1, "half_order_up") == 1
        assert _get_dist_nodata_spread(1, "half_order_down") == 0
        assert _get_dist_nodata_spread(1, 5) == 5

        assert _get_dist_nodata_spread(3, "half_order_up") == 2
        assert _get_dist_nodata_spread(3, "half_order_down") == 1
        assert _get_dist_nodata_spread(3, 5) == 5

        assert _get_dist_nodata_spread(5, "half_order_up") == 3
        assert _get_dist_nodata_spread(5, "half_order_down") == 2
        assert _get_dist_nodata_spread(5, 5) == 5

    @pytest.mark.parametrize(
        "method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    )  # type: ignore
    def test_interpn_interpolator_accuracy(
        self, method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    ):
        """Test that _interpn_interpolator (used by interp_points) returns exactly the same result as scipy.interpn."""

        # Create synthetic 2D array with non-aligned coordinates in X/Y, and X in descending order to mirror a raster's
        # coordinates
        shape = (50, 20)
        coords = (np.linspace(10, 0, shape[0]), np.linspace(20, 30, shape[1]))
        values = np.random.default_rng(42).normal(size=shape)

        # Get 10 random points in the array boundaries
        i = np.random.default_rng(42).uniform(coords[0][-1], coords[0][0], size=10)
        j = np.random.default_rng(42).uniform(coords[1][0], coords[1][-1], size=10)

        # Compare interpn and interpolator

        # Method splinef2d is expecting strictly ascending coordinates (while other methods support desc or asc)
        if method != "splinef2d":
            vals = interpn(points=coords, values=values, xi=(i, j), method=method)
        else:
            vals = interpn(
                points=(np.flip(coords[0]), coords[1]), values=np.flip(values[:], axis=0), xi=(i, j), method=method
            )
        # With the interpolator (coordinates are re-ordered automatically, as it happens often for rasters)
        interpolator = _interpn_interpolator(points=coords, values=values, method=method)
        vals2 = interpolator((i, j))

        assert np.array_equal(vals, vals2, equal_nan=True)

    @pytest.mark.parametrize("tag_aop", [None, "Area", "Point"])  # type: ignore
    @pytest.mark.parametrize("shift_aop", [True, False])  # type: ignore
    def test_interp_points__synthetic(self, tag_aop: Literal["Area", "Point"] | None, shift_aop: bool) -> None:
        """
        Test interp_points function with synthetic data:

        We select known points and compare to the expected interpolation results across all methods, and all pixel
        interpretations (area_or_point, with or without shift).

        The synthetic data is a 3x3 array of values, which we interpolate at different synthetic points:
        1/ Points falling right on grid coordinates are compared to their value in the grid,
        2/ Points falling halfway between grid coordinates are compared to the mean of the surrounding values in the
        grid (as it should equal the linear interpolation on an equal grid),
        3/ Random points in the grid, compared between methods (forcing to use either scipy.ndimage.map_coordinates or
        scipy.interpolate.interpn under-the-hood, to ensure results are consistent).

        These tests also check the behaviour when returning interpolated for points outside of valid values.
        """

        # We flip the array up/down to facilitate index comparison of Y axis
        arr = np.flipud(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape((3, 3)))
        transform = rio.transform.from_bounds(0, 0, 3, 3, 3, 3)
        raster = gu.Raster.from_array(data=arr, transform=transform, crs=None, nodata=-9999)

        # Define the AREA_OR_POINT attribute without re-transforming
        raster.set_area_or_point(tag_aop, shift_area_or_point=False)

        # Check interpolation falls right on values for points (1, 1), (1, 2) etc...
        index_x = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        index_y = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        # The actual X/Y coords will be offset by one because Y axis is inverted and pixel coords is upper-left corner
        points_x, points_y = raster.ij2xy(i=index_x, j=index_y, shift_area_or_point=shift_aop)

        # The following 4 methods should yield the same result because:
        # Nearest = Linear interpolation at the location of a data point
        # Regular grid = Equal grid interpolation at the location of a data point

        raster_points = raster.interp_points(
            (points_x, points_y), method="nearest", shift_area_or_point=shift_aop, as_array=True
        )
        raster_points_lin = raster.interp_points(
            (points_x, points_y), method="linear", shift_area_or_point=shift_aop, as_array=True
        )
        raster_points_interpn = raster.interp_points(
            (points_x, points_y),
            method="nearest",
            force_scipy_function="interpn",
            shift_area_or_point=shift_aop,
            as_array=True,
        )
        raster_points_interpn_lin = raster.interp_points(
            (points_x, points_y),
            method="linear",
            force_scipy_function="interpn",
            shift_area_or_point=shift_aop,
            as_array=True,
        )

        assert np.array_equal(raster_points, raster_points_lin)
        assert np.array_equal(raster_points, raster_points_interpn)
        assert np.array_equal(raster_points, raster_points_interpn_lin)

        for i in range(3):
            for j in range(3):
                ind = 3 * i + j
                assert raster_points[ind] == arr[index_x[ind], index_y[ind]]

        # Check bilinear interpolation values inside the grid (same here, offset by 1 between X and Y)
        index_x_in = [0.5, 0.5, 1.5, 1.5]
        index_y_in = [0.5, 1.5, 0.5, 1.5]

        points_x_in, points_y_in = raster.ij2xy(i=index_x_in, j=index_y_in, shift_area_or_point=shift_aop)

        # Here again compare methods
        raster_points_in = raster.interp_points(
            (points_x_in, points_y_in), method="linear", shift_area_or_point=shift_aop, as_array=True
        )
        raster_points_in_interpn = raster.interp_points(
            (points_x_in, points_y_in),
            method="linear",
            force_scipy_function="interpn",
            shift_area_or_point=shift_aop,
            as_array=True,
        )

        assert np.array_equal(raster_points_in, raster_points_in_interpn)

        for i in range(len(points_x_in)):
            xlow = int(index_x_in[i] - 0.5)
            xupp = int(index_x_in[i] + 0.5)
            ylow = int(index_y_in[i] - 0.5)
            yupp = int(index_y_in[i] + 0.5)

            # Check the bilinear interpolation matches the mean value of those 4 points (equivalent as its the middle)
            assert raster_points_in[i] == np.mean([arr[xlow, ylow], arr[xupp, ylow], arr[xupp, yupp], arr[xlow, yupp]])

        # Check bilinear extrapolation for points at 1 spacing outside from the input grid
        points_out = (
            [(-1, i) for i in np.arange(1, 4)]
            + [(i, -1) for i in np.arange(1, 4)]
            + [(4, i) for i in np.arange(1, 4)]
            + [(i, 4) for i in np.arange(4, 1)]
        )
        points_out_xy = list(zip(*points_out))
        raster_points_out = raster.interp_points(points_out_xy, as_array=True)
        assert all(~np.isfinite(raster_points_out))

        # To use cubic or quintic, we need a larger grid (minimum 6x6, but let's aim bigger with 50x50)
        arr = np.flipud(np.arange(1, 2501).reshape((50, 50)))
        transform = rio.transform.from_bounds(0, 0, 50, 50, 50, 50)
        raster = gu.Raster.from_array(data=arr, transform=transform, crs=None, nodata=-9999)
        raster.set_area_or_point(tag_aop, shift_area_or_point=False)

        # For this, get random points
        rng = np.random.default_rng(42)
        index_x_in_rand = rng.integers(low=8, high=42, size=(10,)) + rng.normal(scale=0.3)
        index_y_in_rand = rng.integers(low=8, high=42, size=(10,)) + rng.normal(scale=0.3)
        points_x_rand, points_y_rand = raster.ij2xy(i=index_x_in_rand, j=index_y_in_rand, shift_area_or_point=shift_aop)

        for method in ["nearest", "linear"]:
            raster_points_mapcoords = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="map_coordinates",
                shift_area_or_point=shift_aop,
                as_array=True,
            )
            raster_points_interpn = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="interpn",
                shift_area_or_point=shift_aop,
                as_array=True,
            )

            # Not exactly equal in floating point precision since changes in Scipy 1.13.0,
            # see https://github.com/GlacioHack/geoutils/issues/533
            assert np.allclose(raster_points_mapcoords, raster_points_interpn)

        # Check that, outside the edge, the interpolation fails and returns a NaN
        index_x_edge_rand = [-0.5, -0.5, -0.5, 25, 25, 49.5, 49.5, 49.5]
        index_y_edge_rand = [-0.5, 25, 49.5, -0.5, 49.5, -0.5, 25, 49.5]

        points_x_rand, points_y_rand = raster.ij2xy(
            i=index_x_edge_rand, j=index_y_edge_rand, shift_area_or_point=shift_aop
        )

        # Test across all methods
        for method in ["nearest", "linear", "cubic", "quintic"]:
            raster_points_mapcoords_edge = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="map_coordinates",
                shift_area_or_point=shift_aop,
                as_array=True,
            )
            raster_points_interpn_edge = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="interpn",
                shift_area_or_point=shift_aop,
                as_array=True,
            )

            assert all(~np.isfinite(raster_points_mapcoords_edge))
            assert all(~np.isfinite(raster_points_interpn_edge))

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    @pytest.mark.parametrize(
        "method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    )  # type: ignore
    def test_interp_points__real(
        self, example: str, method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    ) -> None:
        """
        Test interp_points for real data, checking in particular the propagation of nodata.

        For a random point (dimension 0) and a group of random points (dimension 1) in a real raster, we check the
        consistency of the output forcing to use either scipy.ndimage.map_coordinates or scipy.interpolate.interpn
        under-the-hood, or returning a regular-grid interpolator.
        """

        # Check the accuracy of the interpolation at an exact point, and between methods

        # Open and crop for speed
        r = gu.Raster(example)
        r = r.crop((r.bounds.left, r.bounds.bottom, r.bounds.left + r.res[0] * 50, r.bounds.bottom + r.res[1] * 50))
        r.set_area_or_point("Area", shift_area_or_point=False)

        # 1/ Test for an individual point (shape can be tricky in 1 dimension)
        itest = 10
        jtest = 10
        x, y = r.ij2xy(itest, jtest)
        val = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates", as_array=True)[0]
        val_img = r.get_nanarray()[itest, jtest]
        # For a point exactly at a grid coordinate, only nearest and linear will match
        # (cubic modifies values at a grid coordinate)
        if method in ["nearest", "linear"]:
            assert val_img == pytest.approx(val, nan_ok=True)

        # Check the result is exactly the same for both methods
        val2 = r.interp_points((x, y), method=method, force_scipy_function="interpn", as_array=True)[0]
        assert val2 == pytest.approx(val, nan_ok=True)

        # Check that interp convert to latlon
        lat, lon = gu.projtools.reproject_to_latlon([x, y], in_crs=r.crs)
        val_latlon = r.interp_points((lat, lon), method=method, input_latlon=True, as_array=True)[0]
        assert val == pytest.approx(val_latlon, abs=0.0001, nan_ok=True)

        # 2/ Test for multiple points
        i = np.random.default_rng(42).integers(1, 49, size=10)
        j = np.random.default_rng(42).integers(1, 49, size=10)
        x, y = r.ij2xy(i, j)
        vals = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates", as_array=True)
        vals2 = r.interp_points((x, y), method=method, force_scipy_function="interpn", as_array=True)

        assert np.array_equal(vals, vals2, equal_nan=True)

        # 3/ Test return_interpolator is consistent with above
        interp = _interp_points(
            r.get_nanarray(),
            transform=r.transform,
            area_or_point=r.area_or_point,
            points=(x, y),
            method=method,
            return_interpolator=True,
        )
        vals3 = interp((y, x))
        vals3 = np.array(np.atleast_1d(vals3), dtype=np.float32)

        assert np.array_equal(vals2, vals3, equal_nan=True)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    @pytest.mark.parametrize(
        "method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    )  # type: ignore
    @pytest.mark.parametrize("dist", ["half_order_up", "half_order_down", 0, 1, 5])  # type: ignore
    def test_interp_point__nodata_propag(
        self,
        example: str,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"],
        dist: Literal["half_order_up", "half_order_down"] | int,
    ) -> None:
        """
        Tests for nodata propagation of interp_points.

        We create artificial NaNs at certain pixels of real rasters (one with integer type, one floating type), then
        verify that interpolated values propagate these NaNs at the right distances for all methods, and across all
        nodata propagation distances.

        We also assess the precision of the interpolation under the default nodata propagation "half_order_up", by
        comparing interpolated values near nodata edges with and without the gap-filling of a constant placeholder
        value (= ensures the gap-filling of the array required for most methods that do not support NaN has little
        effect of interpolation).
        """
        # Open and crop for speed
        r = gu.Raster(example)
        r = r.crop((r.bounds.left, r.bounds.bottom, r.bounds.left + r.res[0] * 100, r.bounds.bottom + r.res[1] * 100))

        # 1/ Check the propagation of NaNs

        # 1.1/ Manual check for a specific point
        # Convert raster to float
        r = r.astype(np.float32)

        # Create a NaN at a given pixel (we know the landsat example has no NaNs to begin with)
        i0, j0 = (30, 30)  # This needs to be an area without NaN (and surroundings) in all test example
        r[i0, j0] = np.nan

        # Create a big NaN area in the middle (for more complex NaN propagation below)
        r[40:50, 40:50] = np.nan

        # All surrounding pixels with distance half the method order rounded up should be NaNs
        order = method_to_order[method]
        d = _get_dist_nodata_spread(order=order, dist_nodata_spread=dist)

        # Get indices of raster pixels within the right distance from NaNs
        indices_nan = [
            (i0 + i, j0 + j) for i in np.arange(-d, d + 1) for j in np.arange(-d, d + 1) if (np.abs(i) + np.abs(j)) <= d
        ]
        i, j = list(zip(*indices_nan))
        x, y = r.ij2xy(i, j)
        vals = r.interp_points(
            (x, y), method=method, force_scipy_function="map_coordinates", dist_nodata_spread=dist, as_array=True
        )
        vals2 = r.interp_points(
            (x, y), method=method, force_scipy_function="interpn", dist_nodata_spread=dist, as_array=True
        )

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # Same check for random coordinates within half a pixel of the above coordinates (falling exactly on grid
        # points)
        xoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))
        yoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))

        vals = r.interp_points(
            (x + xoffset, y + yoffset),
            method=method,
            force_scipy_function="map_coordinates",
            dist_nodata_spread=dist,
            as_array=True,
        )
        vals2 = r.interp_points(
            (x + xoffset, y + yoffset),
            method=method,
            force_scipy_function="interpn",
            dist_nodata_spread=dist,
            as_array=True,
        )

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # 1.2/ Check for all NaNs in the raster

        # We create the mask of dilated NaNs
        mask_nan = ~np.isfinite(r.get_nanarray())
        if d != 0:
            mask_nan_dilated = binary_dilation(mask_nan, iterations=d).astype("uint8")
        # (Zero iteration triggers a different behaviour than just "doing nothing" in binary_dilation, we override here)
        else:
            mask_nan_dilated = mask_nan.astype("uint8")
        # Get indices of the related pixels, convert to coordinates
        i, j = np.where(mask_nan_dilated)
        x, y = r.ij2xy(i, j)
        # And interpolate at those coordinates
        vals = r.interp_points(
            (x, y), method=method, force_scipy_function="map_coordinates", dist_nodata_spread=dist, as_array=True
        )
        vals2 = r.interp_points(
            (x, y), method=method, force_scipy_function="interpn", dist_nodata_spread=dist, as_array=True
        )

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # 2/ Check that interpolated values at the edge of NaNs are valid + have small errors due to filling NaNs
        # with the nearest value during interpolation (thanks to the spreading of the nodata mask)

        # We compare values interpolated right at the edge of valid values near a NaN between
        # a/ Implementation of interp_points (that replaces NaNs by the nearest neighbour during interpolation)
        # b/ Raster filled with placeholder value then running interp_points

        # 2.1/ Manual check for a specific point

        # We get the indexes of valid pixels just at the edge of NaNs
        indices_edge = [
            (i0 + i, j0 + j)
            for i in np.arange(-d - 1, d + 2)
            for j in np.arange(-d - 1, d + 2)
            if (np.abs(i) + np.abs(j)) == d + 1
        ]
        i, j = list(zip(*indices_edge))
        x, y = r.ij2xy(i, j)
        # And get their interpolated value
        vals = r.interp_points(
            (x, y), method=method, force_scipy_function="map_coordinates", dist_nodata_spread=dist, as_array=True
        )
        vals2 = r.interp_points(
            (x, y), method=method, force_scipy_function="interpn", dist_nodata_spread=dist, as_array=True
        )

        # Then we fill the NaNs in the raster with a placeholder value of the raster mean
        r_arr = r.get_nanarray()
        r_arr[~np.isfinite(r_arr)] = np.nanmean(r_arr)
        r2 = r.copy(new_array=r_arr)

        # All raster values should be valid now
        assert np.all(np.isfinite(r_arr))

        # Only check the accuracy with the default NaN spreading (half-order rounded up), otherwise anything can happen
        if dist == "half_order_up":

            # Get the interpolated values
            vals_near = r2.interp_points(
                (x, y), method=method, force_scipy_function="map_coordinates", dist_nodata_spread=dist, as_array=True
            )
            vals2_near = r2.interp_points(
                (x, y), method=method, force_scipy_function="interpn", dist_nodata_spread=dist, as_array=True
            )

            # Both sets of values should be valid + within a relative tolerance of 0.01%
            assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-4)
            assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-4)

            # Same check for values within one pixel of the exact coordinates
            xoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))
            yoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))

            vals = r.interp_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="map_coordinates",
                dist_nodata_spread=dist,
                as_array=True,
            )
            vals2 = r.interp_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="interpn",
                dist_nodata_spread=dist,
                as_array=True,
            )
            vals_near = r2.interp_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="map_coordinates",
                dist_nodata_spread=dist,
                as_array=True,
            )
            vals2_near = r2.interp_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="interpn",
                dist_nodata_spread=dist,
                as_array=True,
            )

            # Both sets of values should be exactly the same, without any NaNs
            assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-4)
            assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-4)

            # 2.2/ Repeat the same for all edges of NaNs in the raster
            mask_dilated_plus_one = binary_dilation(mask_nan_dilated, iterations=1).astype(bool)
            mask_edge_dilated = np.logical_and(mask_dilated_plus_one, ~mask_nan_dilated.astype(bool))

            # Get indices of the related pixels, convert to coordinates
            i, j = np.where(mask_edge_dilated)
            x, y = r.ij2xy(i, j)
            # And interpolate at those coordinates
            vals = r.interp_points(
                (x, y), method=method, force_scipy_function="map_coordinates", dist_nodata_spread=dist, as_array=True
            )
            vals2 = r.interp_points(
                (x, y), method=method, force_scipy_function="interpn", dist_nodata_spread=dist, as_array=True
            )
            vals_near = r2.interp_points(
                (x, y), method=method, force_scipy_function="map_coordinates", dist_nodata_spread=dist, as_array=True
            )
            vals2_near = r2.interp_points(
                (x, y), method=method, force_scipy_function="interpn", dist_nodata_spread=dist, as_array=True
            )

            # Both sets of values should be exactly the same, without any NaNs
            assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-4)
            assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-4)

    def test_reduce_points(self) -> None:
        """
        Test reduce points.
        """

        # -- Tests 1: Check based on indexed values --

        # Open raster
        r = gu.Raster(self.landsat_b4_crop_path)

        # A pixel center where all neighbouring coordinates are different:
        # array([[[237, 194, 239],
        #          [250, 173, 164],
        #          [255, 192, 128]]]
        itest0 = 19
        jtest0 = 21

        # Get coordinates at indices
        xtest0, ytest0 = r.ij2xy(itest0, jtest0, force_offset="center")

        # Check that the value at this coordinate is the same as when indexing
        z_val = r.reduce_points((xtest0, ytest0), as_array=True)
        z = r.data.data[itest0, jtest0]
        assert z == z_val

        # Check that the value is the same the other 4 corners of the pixel
        assert z == r.reduce_points((xtest0 + 0.49 * r.res[0], ytest0 - 0.49 * r.res[1]), as_array=True)
        assert z == r.reduce_points((xtest0 - 0.49 * r.res[0], ytest0 + 0.49 * r.res[1]), as_array=True)
        assert z == r.reduce_points((xtest0 - 0.49 * r.res[0], ytest0 - 0.49 * r.res[1]), as_array=True)
        assert z == r.reduce_points((xtest0 + 0.49 * r.res[0], ytest0 + 0.49 * r.res[1]), as_array=True)

        # -- Tests 2: check arguments work as intended --

        # 1/ Lat-lon argument check by getting the coordinates of our last test point
        lat, lon = reproject_to_latlon(points=[xtest0, ytest0], in_crs=r.crs)
        z_val_2 = r.reduce_points((lon, lat), input_latlon=True, as_array=True)
        assert z_val == z_val_2

        # 2/ Band argument
        # Get the band indexes for the multi-band Raster
        r_multi = gu.Raster(self.landsat_rgb_path)
        itest, jtest = r_multi.xy2ij(xtest0, ytest0)
        itest = int(itest[0])
        jtest = int(jtest[0])
        # Extract the values
        z_band1 = r_multi.reduce_points((xtest0, ytest0), band=1, as_array=True)
        z_band2 = r_multi.reduce_points((xtest0, ytest0), band=2, as_array=True)
        z_band3 = r_multi.reduce_points((xtest0, ytest0), band=3, as_array=True)
        # Compare to the Raster array slice
        assert list(r_multi.data[:, itest, jtest]) == [z_band1, z_band2, z_band3]

        # 3/ Masked argument
        r_multi.data[:, itest, jtest] = np.ma.masked
        z_not_ma = r_multi.reduce_points((xtest0, ytest0), band=1, as_array=True)
        assert not np.ma.is_masked(z_not_ma)
        z_ma = r_multi.reduce_points((xtest0, ytest0), band=1, masked=True, as_array=True)
        assert np.ma.is_masked(z_ma)

        # 4/ Window argument
        val_window, z_window = r_multi.reduce_points(
            (xtest0, ytest0), band=1, window=3, masked=True, return_window=True, as_array=True
        )
        assert (
            val_window
            == np.ma.mean(r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])
            == np.ma.mean(z_window)
        )
        assert np.array_equal(z_window, r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])

        # 5/ Reducer function argument
        val_window2 = r_multi.reduce_points(
            (xtest0, ytest0), band=1, window=3, masked=True, reducer_function=np.ma.median, as_array=True
        )
        assert val_window2 == np.ma.median(r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])

        # -- Tests 3: check that errors are raised when supposed for non-boolean arguments --

        # Verify that passing a window that is not a whole number fails
        with pytest.raises(ValueError, match=re.escape("Window must be a whole number.")):
            r.reduce_points((xtest0, ytest0), window=3.5)  # type: ignore
        # Same for an odd number
        with pytest.raises(ValueError, match=re.escape("Window must be an odd number.")):
            r.reduce_points((xtest0, ytest0), window=4)
        # But a window that is a whole number as a float works
        r.reduce_points((xtest0, ytest0), window=3.0)  # type: ignore

        # -- Tests 4: check that passing an array-like object works

        # For simple coordinates
        x_coords = [xtest0, xtest0 + 10]
        y_coords = [ytest0, ytest0 - 10]
        vals = r_multi.reduce_points((x_coords, y_coords), as_array=True)
        val0, win0 = r_multi.reduce_points((x_coords[0], y_coords[0]), return_window=True, as_array=True)
        val1, win1 = r_multi.reduce_points((x_coords[1], y_coords[1]), return_window=True, as_array=True)

        assert len(vals) == len(x_coords)
        assert np.array_equal(vals[0], val0, equal_nan=True)
        assert np.array_equal(vals[1], val1, equal_nan=True)

        # With a return window argument
        vals, windows = r_multi.reduce_points((x_coords, y_coords), return_window=True, as_array=True)
        assert len(windows) == len(x_coords)
        assert np.array_equal(windows[0], win0, equal_nan=True)
        assert np.array_equal(windows[1], win1, equal_nan=True)

        # -- Tests 5 -- Check image corners and latlon argument

        # Lower right pixel
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lat, lon = reproject_to_latlon([x, y], r.crs)
        assert (
            r.reduce_points((x, y), as_array=True)
            == r.reduce_points((lon, lat), input_latlon=True, as_array=True)
            == r.data[-1, -1]
        )

        # One pixel above
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + 3 * r.res[1] / 2]
        lat, lon = reproject_to_latlon([x, y], r.crs)
        assert (
            r.reduce_points((x, y), as_array=True)
            == r.reduce_points((lon, lat), input_latlon=True, as_array=True)
            == r.data[-2, -1]
        )

        # One pixel left
        x, y = [r.bounds.right - 3 * r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lat, lon = reproject_to_latlon([x, y], r.crs)
        assert (
            r.reduce_points((x, y), as_array=True)
            == r.reduce_points((lon, lat), input_latlon=True, as_array=True)
            == r.data[-1, -2]
        )
