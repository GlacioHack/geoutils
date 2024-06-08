from __future__ import annotations

import re
from typing import Literal
import pytest

import numpy as np
import rasterio as rio
import geoutils as gu
from scipy.ndimage import distance_transform_edt, binary_dilation

from geoutils.raster.interpolate import method_to_order
from geoutils.projtools import reproject_to_latlon
from geoutils import examples

class TestInterpolate:

    landsat_b4_path = examples.get_path("everest_landsat_b4")
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    landsat_b4_crop_path = examples.get_path("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")

    @pytest.mark.parametrize("tag_aop", [None, "Area", "Point"])  # type: ignore
    @pytest.mark.parametrize("shift_aop", [True, False])  # type: ignore
    def test_interp_points__synthetic(self, tag_aop: str | None, shift_aop: bool) -> None:
        """Test interp_points function with synthetic data."""

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

        raster_points = raster.interp_points((points_x, points_y), method="nearest", shift_area_or_point=shift_aop)
        raster_points_lin = raster.interp_points((points_x, points_y), method="linear", shift_area_or_point=shift_aop)
        raster_points_interpn = raster.interp_points(
            (points_x, points_y), method="nearest", force_scipy_function="interpn", shift_area_or_point=shift_aop
        )
        raster_points_interpn_lin = raster.interp_points(
            (points_x, points_y), method="linear", force_scipy_function="interpn", shift_area_or_point=shift_aop
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
            (points_x_in, points_y_in), method="linear", shift_area_or_point=shift_aop
        )
        raster_points_in_interpn = raster.interp_points(
            (points_x_in, points_y_in), method="linear", force_scipy_function="interpn", shift_area_or_point=shift_aop
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
        raster_points_out = raster.interp_points(points_out_xy)
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
            )
            raster_points_interpn = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="interpn",
                shift_area_or_point=shift_aop,
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

        # Nearest doesn't apply, just linear and above
        for method in ["cubic", "quintic"]:
            raster_points_mapcoords_edge = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="map_coordinates",
                shift_area_or_point=shift_aop,
            )
            raster_points_interpn_edge = raster.interp_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="interpn",
                shift_area_or_point=shift_aop,
            )

            assert all(~np.isfinite(raster_points_mapcoords_edge))
            assert all(~np.isfinite(raster_points_interpn_edge))

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    @pytest.mark.parametrize(
        "method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    )  # type: ignore
    def test_interp_points__real(
            self, example: str,
            method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    ) -> None:
        """Test interp_points for real data."""

        # 1/ Check the accuracy of the interpolation at an exact point, and between methods

        # Open and crop for speed
        r = gu.Raster(example)
        r = r.crop((r.bounds.left, r.bounds.bottom, r.bounds.left + r.res[0] * 50, r.bounds.bottom + r.res[1] * 50))
        r.set_area_or_point("Area", shift_area_or_point=False)

        # Test for an individual point (shape can be tricky at 1 dimension)
        itest = 10
        jtest = 10
        x, y = r.ij2xy(itest, jtest)
        val = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates")[0]
        val_img = r.data[itest, jtest]
        if "nearest" in method or "linear" in method:
            assert val_img == val

        # Check the result is exactly the same for both methods
        val2 = r.interp_points((x, y), method=method, force_scipy_function="interpn")[0]
        assert val2 == pytest.approx(val)

        # Check that interp convert to latlon
        lat, lon = gu.projtools.reproject_to_latlon([x, y], in_crs=r.crs)
        val_latlon = r.interp_points((lat, lon), method=method, input_latlon=True)[0]
        assert val == pytest.approx(val_latlon, abs=0.0001)

        # 2/ Check the propagation of NaNs

        # 2.1/ Manual check for a specific point
        # Convert raster to float
        r = r.astype(np.float32)

        # Create a NaN at a given pixel (we know the landsat example has no NaNs to begin with)
        i0, j0 = (10, 10)
        r[i0, j0] = np.nan

        # All surrounding pixels with distance half the method order rounded up should be NaNs
        order = method_to_order[method]
        d = int(np.ceil(order/2))
        if method in ["nearest", "linear"]:
            return
        # Get indices of NaNs within the distance from NaNs
        indices_nan = [
            (i0 + i, j0 + j) for i in np.arange(-d, d + 1) for j in np.arange(-d, d + 1) if (np.abs(i) + np.abs(j)) <= d
        ]
        i, j = list(zip(*indices_nan))
        x, y = r.ij2xy(i, j)
        vals = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates")
        vals2 = r.interp_points((x, y), method=method, force_scipy_function="interpn")

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # Same check for values within one pixel of the exact coordinates
        xoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))
        yoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))

        vals = r.interp_points((x + xoffset, y + yoffset), method=method, force_scipy_function="map_coordinates")
        vals2 = r.interp_points((x + xoffset, y + yoffset), method=method, force_scipy_function="interpn")

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # 2.2/ Check for all NaNs in the raster

        # We create the mask of dilated NaNs
        mask_nan = ~np.isfinite(r.get_nanarray())
        mask_nan_dilated = binary_dilation(mask_nan, iterations=d).astype("uint8")
        # Get indices of the related pixels, convert to coordinates
        i, j = np.where(mask_nan_dilated)
        x, y = r.ij2xy(i, j)
        # And interpolate at those coordinates
        vals = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates")
        vals2 = r.interp_points((x, y), method=method, force_scipy_function="interpn")

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # 3/ Check that valid interpolated values at the edge of NaNs have small errors caused by filling NaNs
        # with the nearest value during interpolation (thanks to the spreading of the nodata mask)

        # We compare values interpolated right at the edge of valid values near a NaN between
        # 1/ Implementation of interp_points (that replaces NaNs by the nearest neighbour during interpolation)
        # 2/ Raster filled with placeholder value then running interp_points

        # 3.1/ Manual check for a specific point

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
        vals = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates")
        vals2 = r.interp_points((x, y), method=method, force_scipy_function="interpn")

        # Then we fill the NaNs in the raster with a placeholder value of the DEM mean
        r_arr = r.get_nanarray()
        r_arr[~np.isfinite(r_arr)] = np.nanmean(r_arr)
        r2 = r.copy(new_array=r_arr)

        # All raster values should be valid now
        assert np.all(np.isfinite(r_arr))

        # And get the interpolated values
        vals_near = r2.interp_points((x, y), method=method, force_scipy_function="map_coordinates")
        vals2_near = r2.interp_points((x, y), method=method, force_scipy_function="interpn")

        # Both sets of values should be exactly the same, without any NaNs
        assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-2)
        assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-2)

        # Same check for values within one pixel of the exact coordinates
        # Same check for values within one pixel of the exact coordinates
        xoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))
        yoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))

        vals = r.interp_points((x + xoffset, y + yoffset), method=method, force_scipy_function="map_coordinates")
        vals2 = r.interp_points((x + xoffset, y + yoffset), method=method, force_scipy_function="interpn")
        vals_near = r2.interp_points((x + xoffset, y + yoffset), method=method, force_scipy_function="map_coordinates")
        vals2_near = r2.interp_points((x + xoffset, y + yoffset), method=method, force_scipy_function="interpn")

        # Both sets of values should be exactly the same, without any NaNs
        assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-2)
        assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-2)

        # 3.2/ Repeat the same for all NaNs
        mask_dilated_plus_one = binary_dilation(mask_nan_dilated, iterations=1).astype(bool)
        mask_edge_dilated = np.logical_and(mask_dilated_plus_one, ~mask_nan_dilated.astype(bool))

        # Get indices of the related pixels, convert to coordinates
        i, j = np.where(mask_edge_dilated)
        x, y = r.ij2xy(i, j)
        # And interpolate at those coordinates
        vals = r.interp_points((x, y), method=method, force_scipy_function="map_coordinates")
        vals2 = r.interp_points((x, y), method=method, force_scipy_function="interpn")
        vals_near = r2.interp_points((x, y), method=method, force_scipy_function="map_coordinates")
        vals2_near = r2.interp_points((x, y), method=method, force_scipy_function="interpn")

        # Both sets of values should be exactly the same, without any NaNs
        assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-2)
        assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-2)

    def test_value_at_coords(self) -> None:
        """
        Test that value at coords works as intended
        """

        # -- Tests 1: check based on indexed values --

        # Open raster
        r = gu.Raster(self.landsat_b4_crop_path)

        # A pixel center where all neighbouring coordinates are different:
        # array([[[237, 194, 239],
        #          [250, 173, 164],
        #          [255, 192, 128]]]
        itest0 = 120
        jtest0 = 451
        # This is the center of the pixel
        xtest0 = 496975.0
        ytest0 = 3099095.0

        # Verify coordinates match indexes
        x_out, y_out = r.ij2xy(itest0, jtest0, force_offset="center")
        assert x_out == xtest0
        assert y_out == ytest0

        # Check that the value at this coordinate is the same as when indexing
        z_val = r.value_at_coords(xtest0, ytest0)
        z = r.data.data[itest0, jtest0]
        assert z == z_val

        # Check that the value is the same the other 4 corners of the pixel
        assert z == r.value_at_coords(xtest0 + 0.49 * r.res[0], ytest0 - 0.49 * r.res[1])
        assert z == r.value_at_coords(xtest0 - 0.49 * r.res[0], ytest0 + 0.49 * r.res[1])
        assert z == r.value_at_coords(xtest0 - 0.49 * r.res[0], ytest0 - 0.49 * r.res[1])
        assert z == r.value_at_coords(xtest0 + 0.49 * r.res[0], ytest0 + 0.49 * r.res[1])

        # -- Tests 2: check arguments work as intended --

        # 1/ Lat-lon argument check by getting the coordinates of our last test point
        lat, lon = reproject_to_latlon(points=[xtest0, ytest0], in_crs=r.crs)
        z_val_2 = r.value_at_coords(lon, lat, latlon=True)
        assert z_val == z_val_2

        # 2/ Band argument
        # Get the band indexes for the multi-band Raster
        r_multi = gu.Raster(self.landsat_rgb_path)
        itest, jtest = r_multi.xy2ij(xtest0, ytest0)
        itest = int(itest[0])
        jtest = int(jtest[0])
        # Extract the values
        z_band1 = r_multi.value_at_coords(xtest0, ytest0, band=1)
        z_band2 = r_multi.value_at_coords(xtest0, ytest0, band=2)
        z_band3 = r_multi.value_at_coords(xtest0, ytest0, band=3)
        # Compare to the Raster array slice
        assert list(r_multi.data[:, itest, jtest]) == [z_band1, z_band2, z_band3]

        # 3/ Masked argument
        r_multi.data[:, itest, jtest] = np.ma.masked
        z_not_ma = r_multi.value_at_coords(xtest0, ytest0, band=1)
        assert not np.ma.is_masked(z_not_ma)
        z_ma = r_multi.value_at_coords(xtest0, ytest0, band=1, masked=True)
        assert np.ma.is_masked(z_ma)

        # 4/ Window argument
        val_window, z_window = r_multi.value_at_coords(
            xtest0, ytest0, band=1, window=3, masked=True, return_window=True
        )
        assert (
                val_window
                == np.ma.mean(r_multi.data[0, itest - 1: itest + 2, jtest - 1: jtest + 2])
                == np.ma.mean(z_window)
        )
        assert np.array_equal(z_window, r_multi.data[0, itest - 1: itest + 2, jtest - 1: jtest + 2])

        # 5/ Reducer function argument
        val_window2 = r_multi.value_at_coords(
            xtest0, ytest0, band=1, window=3, masked=True, reducer_function=np.ma.median
        )
        assert val_window2 == np.ma.median(r_multi.data[0, itest - 1: itest + 2, jtest - 1: jtest + 2])

        # -- Tests 3: check that errors are raised when supposed for non-boolean arguments --

        # Verify that passing a window that is not a whole number fails
        with pytest.raises(ValueError, match=re.escape("Window must be a whole number.")):
            r.value_at_coords(xtest0, ytest0, window=3.5)  # type: ignore
        # Same for an odd number
        with pytest.raises(ValueError, match=re.escape("Window must be an odd number.")):
            r.value_at_coords(xtest0, ytest0, window=4)
        # But a window that is a whole number as a float works
        r.value_at_coords(xtest0, ytest0, window=3.0)  # type: ignore

        # -- Tests 4: check that passing an array-like object works

        # For simple coordinates
        x_coords = [xtest0, xtest0 + 100]
        y_coords = [ytest0, ytest0 - 100]
        vals = r_multi.value_at_coords(x=x_coords, y=y_coords)
        val0, win0 = r_multi.value_at_coords(x=x_coords[0], y=y_coords[0], return_window=True)
        val1, win1 = r_multi.value_at_coords(x=x_coords[1], y=y_coords[1], return_window=True)

        assert len(vals) == len(x_coords)
        assert vals[0] == val0
        assert vals[1] == val1

        # With a return window argument
        vals, windows = r_multi.value_at_coords(x=x_coords, y=y_coords, return_window=True)
        assert len(windows) == len(x_coords)
        assert np.array_equal(windows[0], win0, equal_nan=True)
        assert np.array_equal(windows[1], win1, equal_nan=True)

        # -- Tests 5 -- Check image corners and latlon argument

        # Lower right pixel
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lat, lon = reproject_to_latlon([x, y], r.crs)
        assert r.value_at_coords(x, y) == r.value_at_coords(lon, lat, latlon=True) == r.data[-1, -1]

        # One pixel above
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + 3 * r.res[1] / 2]
        lat, lon = reproject_to_latlon([x, y], r.crs)
        assert r.value_at_coords(x, y) == r.value_at_coords(lon, lat, latlon=True) == r.data[-2, -1]

        # One pixel left
        x, y = [r.bounds.right - 3 * r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lat, lon = reproject_to_latlon([x, y], r.crs)
        assert r.value_at_coords(x, y) == r.value_at_coords(lon, lat, latlon=True) == r.data[-1, -2]