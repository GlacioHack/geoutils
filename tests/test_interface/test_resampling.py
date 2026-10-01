from __future__ import annotations

import os.path
import re
import tempfile
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from affine import Affine

import geoutils as gu
from geoutils import examples, open_raster
from geoutils.interface.resampling import (
    _interpolate_array,
)
from geoutils.multiproc import MultiprocConfig
from geoutils.operators import GridNeighbours, Interpolator, LocalData, PointNeighbours, Reducer
from geoutils.operators.interpolator import (
    InverseDistance,
    Kriging,
    Linear,
    Nearest,
    ScipyInterpolationMethod,
    ScipyInterpolator,
)
from geoutils.operators.nodata import NodataChoice
from geoutils.operators.reducer import Mean, Median, Sum
from geoutils.projtools import reproject_to_latlon
from tests.operator_helpers import (
    LocalMeanInterpolator,
    NoSupportReducer,
    PropagatingLocalMeanInterpolator,
    PropagatingMeanReducer,
    WindowRangeInterpolator,
)


class TestResampling:
    """Test module for point sampling names, coordinates, output formats and option forwarding.

    Method accuracy and numerical engine comparisons are covered in test_operators/test_interpolator.py
    and test_operators/test_reducer.py.
    """

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")
    landsat_b4_crop_path = examples.get_path_test("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")

    def test_at_points__new_names(self) -> None:
        """Checks that the new point sampling names return the expected interpolated value and window mean."""

        # Use a simple grid so the center value and its three by three mean are known
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32606)
        point = raster.ij2xy(2, 2)

        # A Raster can interpolate its center cell and reduce the surrounding window
        interpolated = raster.interp_at_points(point, method="nearest", as_array=True)
        reduced = raster.reduce_at_points(point, window=3, as_array=True)
        accessor_interpolated = raster.to_xarray().rst.interp_at_points(point, method="nearest", as_array=True)

        # The center cell is 12, and the symmetric window has the same mean
        np.testing.assert_array_equal(interpolated, [12])
        np.testing.assert_array_equal(reduced, [12])
        np.testing.assert_array_equal(accessor_interpolated, [12])

    def test_at_points__deprecated_names(self) -> None:
        """Checks that the old point sampling names warn and forward their options to the new methods."""

        # Window median different from its center
        values = np.arange(25, dtype=float).reshape(5, 5)
        values[2, 2] = 100
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32606)
        point = raster.ij2xy(2, 2)

        # Each old name raises one deprecation warning and forwards the same arguments
        expected_interp = raster.interp_at_points(point, method="nearest", as_array=True)
        with pytest.warns(DeprecationWarning, match=r"interp_points\(\).*interp_at_points\(\)"):
            actual_interp = raster.interp_points(point, method="nearest", as_array=True)
        np.testing.assert_array_equal(actual_interp, expected_interp)

        expected_reduce = raster.reduce_at_points(point, window=3, reducer_function=np.ma.median, as_array=True)
        with pytest.warns(DeprecationWarning, match=r"reduce_points\(\).*reduce_at_points\(\)"):
            actual_reduce = raster.reduce_points(point, window=3, reducer_function=np.ma.median, as_array=True)
        np.testing.assert_array_equal(actual_reduce, expected_reduce)

    def test_interpolate_array__band_masks(self) -> None:
        """Handle masked, NaN and infinite values independently in every raster band."""

        # Each band uses a different invalid representation and position
        source = np.ma.array(
            np.stack((np.arange(16, dtype=np.float32).reshape(4, 4), np.arange(16, dtype=np.float32).reshape(4, 4))),
            mask=False,
        )
        source.mask[0, 1, 1] = True
        source[1, 2, 2] = np.inf
        transform = rio.transform.from_origin(0, 4, 1, 1)

        # An unchanged grid must retain finite values and each band's own invalid cell
        actual = _interpolate_array(source, src_transform=transform, dst_transform=transform)
        assert actual.shape == source.shape
        assert np.array_equal(np.isnan(actual[0]), source.mask[0])
        assert np.array_equal(np.isnan(actual[1]), ~np.isfinite(source.data[1]))
        assert np.array_equal(actual[:, 0, 0], source.data[:, 0, 0])

    @pytest.mark.parametrize("shape", [(3, 7), (7, 3)])  # landscape and portrait exercise different bounds axes
    def test_interp_points__nonsquare(self, shape: tuple[int, int]) -> None:
        """
        Regression test: interp_points must not drop valid points on a non-square raster.

        The buggy out-of-bounds mask compared the row index against the number of columns and the column
        index against the number of rows (axes swapped), so points whose index exceeded the smaller raster
        dimension were wrongly flagged as out-of-bounds and returned as NaN. Square rasters hide this, so we
        test both a landscape (rows < cols) and a portrait (rows > cols) raster, which exercise the two axes
        independently.
        """
        nrows, ncols = shape
        # Unique value per pixel so nearest-neighbour sampling is an exact ground-truth check
        arr = (np.arange(nrows)[:, None] * ncols + np.arange(ncols)[None, :]).astype("float32")
        transform = rio.transform.from_bounds(0, 0, ncols, nrows, ncols, nrows)
        raster = gu.Raster.from_array(data=arr, transform=transform, crs=None, nodata=-9999)
        raster.set_area_or_point("Area", shift_area_or_point=False)

        # Sample every pixel centre, plus one clearly out-of-bounds point, in the same call.
        index_i, index_j = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
        x, y = raster.ij2xy(i=index_i.ravel(), j=index_j.ravel(), shift_area_or_point=True)
        x = np.append(x, -5.0)
        y = np.append(y, -5.0)
        vals = raster.interp_at_points((x, y), method="nearest", shift_area_or_point=True, as_array=True)

        # Every valid pixel centre is finite and equals its pixel value; the out-of-bounds point stays NaN
        assert np.all(np.isfinite(vals[:-1]))
        np.testing.assert_allclose(vals[:-1], raster.to_nanarray()[index_i.ravel(), index_j.ravel()])
        assert not np.isfinite(vals[-1])

    @pytest.mark.parametrize("method", ["interp_at_points", "reduce_at_points"])
    @pytest.mark.parametrize("loaded", [False, True], ids=["unloaded", "loaded"])
    def test_interp_reduce_at_points__out_of_bounds(self, method: str, loaded: bool, tmp_path: Path) -> None:
        """Checks that all point sampling names return NaN beyond the bounds of loaded and unloaded rasters."""

        # Create synthetic raster and write to file to test MP
        values = np.arange(1, 31, dtype=np.float32).reshape(5, 6)
        raster = gu.Raster.from_array(
            values, rio.transform.from_origin(500_000, 4_100_000, 10, 20), crs=32610, nodata=0
        )
        filename = tmp_path / "source.tif"
        raster.to_file(filename)
        source = gu.Raster(filename, load_data=loaded)

        # We query two lat/lon corners, slightly out of bounds
        x = np.array([raster.bounds.left - 1_000, raster.bounds.right + 1_000])
        y = np.array([raster.bounds.top + 1_000, raster.bounds.bottom - 1_000])
        longitude, latitude = reproject_to_latlon((x, y), raster.crs)
        options = {"method": "nearest"} if method.startswith("interp") else {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = getattr(source, method)((longitude, latitude), input_latlon=True, as_array=True, **options)

        # Check every outside value is NaN, and source data is not loaded with MP
        np.testing.assert_array_equal(result, [np.nan, np.nan])
        assert source.is_loaded == loaded
        if method.startswith("interp"):
            assert any(
                "All provided points were outside of raster bounds" in str(warning.message) for warning in caught
            )
        if method in {"interp_points", "reduce_points"}:
            assert any(issubclass(warning.category, DeprecationWarning) for warning in caught)

    @pytest.mark.parametrize("method", ["interp_at_points", "reduce_at_points"])
    @pytest.mark.parametrize("point_input_type", ["pointcloud", "accessor", "geodataframe", "latlon"])
    @pytest.mark.parametrize("all_outside", [False, True])
    def test_methods__point_output_coordinates(self, method: str, point_input_type: str, all_outside: bool) -> None:
        """Checks that interpolation and window reduction return points in the raster CRS, including outside points."""

        # Place the first point inside pixel (1, 1) for both sampling methods and the second beyond the grid
        values = np.arange(36, dtype=float).reshape(6, 6)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(500_000, 4_100_000, 10, 10), crs=32610)
        expected_x = np.array([500_012.5, 501_042.5])
        expected_y = np.array([4_099_987.5, 4_099_957.5])
        expected_values = np.array([values[1, 1], np.nan])
        if all_outside:
            expected_x[0] += 1_000
            expected_values[0] = np.nan

        # Express the same query locations as spatial objects or longitude/latitude arrays
        longitude, latitude = reproject_to_latlon((expected_x, expected_y), raster.crs)
        pointcloud = gu.PointCloud.from_xyz(longitude, latitude, np.zeros(2), crs=4326)
        point_inputs = {
            "pointcloud": pointcloud,
            "accessor": pointcloud.ds.pc,
            "geodataframe": pointcloud.ds,
            "latlon": (longitude, latitude),
        }
        point_input = point_inputs[point_input_type]

        # Request point output through the public API; interpolation warns when every query is outside
        sample = getattr(raster, method)
        options = {"method": "nearest"} if method == "interp_at_points" else {}
        if method == "interp_at_points" and all_outside:
            with pytest.warns(UserWarning, match="All provided points were outside of raster bounds"):
                result = sample(point_input, input_latlon=point_input_type == "latlon", **options)
        else:
            result = sample(point_input, input_latlon=point_input_type == "latlon", **options)

        # Check coordinates and values in input order; longitude/latitude conversion rounds to centimeter accuracy
        assert result.crs == raster.crs
        np.testing.assert_allclose(result.geometry.x, expected_x, rtol=0, atol=1e-2)
        np.testing.assert_allclose(result.geometry.y, expected_y, rtol=0, atol=1e-2)
        np.testing.assert_allclose(result.data, expected_values, rtol=0, atol=0, equal_nan=True)

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
        z_val = r.reduce_at_points((xtest0, ytest0), as_array=True)
        z = r.data.data[itest0, jtest0]
        assert z == z_val

        # Check that the value is the same the other 4 corners of the pixel
        assert z == r.reduce_at_points((xtest0 + 0.49 * r.res[0], ytest0 - 0.49 * r.res[1]), as_array=True)
        assert z == r.reduce_at_points((xtest0 - 0.49 * r.res[0], ytest0 + 0.49 * r.res[1]), as_array=True)
        assert z == r.reduce_at_points((xtest0 - 0.49 * r.res[0], ytest0 - 0.49 * r.res[1]), as_array=True)
        assert z == r.reduce_at_points((xtest0 + 0.49 * r.res[0], ytest0 + 0.49 * r.res[1]), as_array=True)

        # -- Tests 2: check arguments work as intended --

        # 1/ Lat-lon argument check by getting the coordinates of our last test point
        lon, lat = reproject_to_latlon(points=(xtest0, ytest0), in_crs=r.crs)
        z_val_2 = r.reduce_at_points((lon, lat), input_latlon=True, as_array=True)
        assert z_val == z_val_2

        # 2/ Band argument
        # Get the band indexes for the multi-band Raster
        r_multi = gu.Raster(self.landsat_rgb_path)
        itest, jtest = r_multi.xy2ij(xtest0, ytest0)
        itest = int(itest[0])
        jtest = int(jtest[0])
        # Extract the values
        z_band1 = r_multi.reduce_at_points((xtest0, ytest0), band=1, as_array=True)
        z_band2 = r_multi.reduce_at_points((xtest0, ytest0), band=2, as_array=True)
        z_band3 = r_multi.reduce_at_points((xtest0, ytest0), band=3, as_array=True)
        # Compare to the Raster array slice
        assert list(r_multi.data[:, itest, jtest]) == [z_band1, z_band2, z_band3]

        # 3/ Masked argument
        r_multi.data[:, itest, jtest] = np.ma.masked
        z_not_ma = r_multi.reduce_at_points((xtest0, ytest0), band=1, as_array=True)
        assert not np.ma.is_masked(z_not_ma)
        z_ma = r_multi.reduce_at_points((xtest0, ytest0), band=1, masked=True, as_array=True)
        assert np.ma.is_masked(z_ma)

        # 4/ Window argument
        val_window = r_multi.reduce_at_points((xtest0, ytest0), band=1, window=3, masked=True, as_array=True)
        assert val_window == np.ma.mean(r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])

        # 5/ Reducer function argument
        val_window2 = r_multi.reduce_at_points(
            (xtest0, ytest0), band=1, window=3, masked=True, reducer_function=np.ma.median, as_array=True
        )
        assert val_window2 == np.ma.median(r_multi.data[0, itest - 1 : itest + 2, jtest - 1 : jtest + 2])

        # -- Tests 3: check that errors are raised when supposed for non-boolean arguments --

        # Verify that passing a window that is not a whole number fails
        with pytest.raises(ValueError, match=re.escape("Window must be a whole number.")):
            r.reduce_at_points((xtest0, ytest0), window=3.5)  # type: ignore
        # Same for an odd number
        with pytest.raises(ValueError, match=re.escape("Window must be an odd number.")):
            r.reduce_at_points((xtest0, ytest0), window=4)
        # But a window that is a whole number as a float works
        r.reduce_at_points((xtest0, ytest0), window=3.0)  # type: ignore

        # -- Tests 4: check that passing an array-like object works

        # For simple coordinates
        x_coords = [xtest0, xtest0 + 10]
        y_coords = [ytest0, ytest0 - 10]
        vals = r_multi.reduce_at_points((x_coords, y_coords), as_array=True)
        val0 = r_multi.reduce_at_points((x_coords[0], y_coords[0]), as_array=True)
        val1 = r_multi.reduce_at_points((x_coords[1], y_coords[1]), as_array=True)

        assert len(vals) == len(x_coords)
        assert np.array_equal(vals[0], val0, equal_nan=True)
        assert np.array_equal(vals[1], val1, equal_nan=True)

        # -- Tests 5 -- Check image corners and latlon argument

        # Lower right pixel
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lon, lat = reproject_to_latlon((x, y), r.crs)
        lr1 = r.reduce_at_points((x, y), as_array=True)
        lr2 = r.reduce_at_points((lon, lat), input_latlon=True, as_array=True)
        lr3 = r.data[-1, -1]
        assert np.array_equal(lr1, lr2, equal_nan=True)
        assert np.array_equal(lr2, lr3, equal_nan=True)

        # One pixel above
        x, y = [r.bounds.right - r.res[0] / 2, r.bounds.bottom + 3 * r.res[1] / 2]
        lon, lat = reproject_to_latlon((x, y), r.crs)
        lra1 = r.reduce_at_points((x, y), as_array=True)
        lra2 = r.reduce_at_points((lon, lat), input_latlon=True, as_array=True)
        lra3 = r.data[-2, -1]
        assert np.array_equal(lra1, lra2, equal_nan=True)
        assert np.array_equal(lra2, lra3, equal_nan=True)

        # One pixel left
        x, y = [r.bounds.right - 3 * r.res[0] / 2, r.bounds.bottom + r.res[1] / 2]
        lon, lat = reproject_to_latlon((x, y), r.crs)
        lrl1 = r.reduce_at_points((x, y), as_array=True)
        lrl2 = r.reduce_at_points((lon, lat), input_latlon=True, as_array=True)
        lrl3 = r.data[-1, -2]
        assert np.array_equal(lrl1, lrl2, equal_nan=True)
        assert np.array_equal(lrl2, lrl3, equal_nan=True)

    def test_reduce_points__propagates_reducer_uncertainty(self) -> None:
        """Checks that requesting uncertainty leaves reduce_points() values unchanged and accounts for shared cells."""

        # Place two three-by-three windows one column apart so six source pixels contribute to both means
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32606)
        x, y = raster.ij2xy(np.array([2, 2]), np.array([2, 3]))
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Request uncertainty through the same public call and compare its first item to the established result
        expected = raster.reduce_at_points((x, y), reducer_function=Mean(), window=3, as_array=True)
        nominal = raster.reduce_at_points(
            reducer_function=Mean(), window=3, as_array=True, points=(x, y), error_structure=source_error
        )
        summary = gu.uncertainty.propagate(
            raster.reduce_at_points,
            error_structure=source_error,
            operation_kwargs={"reducer_function": Mean(), "window": 3, "as_array": True, "points": (x, y)},
            return_covariance=True,
        )
        np.testing.assert_array_equal(nominal, expected)
        np.testing.assert_array_equal(summary.estimate, expected)

        # Each mean uses nine weights of 1/9; the covariance between means comes from their six shared cells
        expected_covariance = np.array([[4 / 9, 24 / 81], [24 / 81, 4 / 9]])
        assert summary.covariance is not None
        np.testing.assert_allclose(summary.covariance, expected_covariance)

    @pytest.mark.parametrize(("method", "expected_variance"), [("nearest", 4.0), ("linear", 1.5625)])
    def test_interp_points__propagates_string_method_uncertainty(self, method: str, expected_variance: float) -> None:
        """
        Checks that requesting uncertainty with method strings leaves values unchanged and uses the interpolation
        weights.
        """

        # Query one quarter pixel below and right of a source center to give bilinear weights 9/16, 3/16, 3/16, 1/16
        values = np.arange(16, dtype=float).reshape(4, 4)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 4, 1, 1), crs=32606)
        center_x, center_y = raster.ij2xy(1, 1)
        point = (center_x + 0.25, center_y - 0.25)
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Compare the first tuple item with the result of interpolation without requesting uncertainty
        expected = raster.interp_at_points(point, method=method, as_array=True)
        nominal = raster.interp_at_points(method=method, as_array=True, points=point, error_structure=source_error)
        summary = gu.uncertainty.propagate(
            raster.interp_at_points,
            error_structure=source_error,
            operation_kwargs={"method": method, "as_array": True, "points": point},
        )
        np.testing.assert_array_equal(nominal, expected)
        np.testing.assert_array_equal(summary.estimate, expected)

        # Multiply the sum of squared interpolation weights by the independent source variance of four
        np.testing.assert_allclose(summary.variance, [expected_variance])


class TestSamplingOperators:
    """Test module for raster interpolation, operator neighborhoods, missing values and output formats."""

    def test_interp_points__built_in_operator_matches_string(self) -> None:
        """Checks that Linear() and its method string use the same fast kernel with bitwise-equal output."""

        # Create fractional targets over a finite raster so every interpolation contribution is deterministic
        values = np.arange(36, dtype=np.float64).reshape(6, 6)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 6, 1, 1), crs=4326)
        points = (np.array([1.2, 2.7, 4.1]), np.array([4.4, 3.2, 1.8]))

        # Compare both public forms after they select the same regular-grid calculation
        expected = raster.interp_at_points(points, method="linear", as_array=True)
        result = raster.interp_at_points(points, method=Linear(), as_array=True)
        np.testing.assert_array_equal(result, expected)

    def test_interp_points__custom_grid_neighbours(self) -> None:
        """Checks that a custom Interpolator receives the expected source window around each raster target."""

        # Select the center of a five by five raster, whose three by three values span 6 through 18
        values = np.arange(25, dtype=np.float64).reshape(5, 5)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)
        x, y = raster.ij2xy(np.array([2]), np.array([2]))

        # The custom method should receive the complete centered source window and return its known range
        operator = WindowRangeInterpolator()
        result = raster.interp_at_points((x, y), method=operator, as_array=True)
        np.testing.assert_array_equal(result, np.array([12.0]))
        assert operator.batch_calls == 1

    def test_interp_points__automatic_and_custom_grid_neighbours(self) -> None:
        """Checks that a custom method uses a default raster window or an explicitly shaped window."""

        # Put one high value in the corner of a five by five window around the requested center
        values = np.zeros((5, 5), dtype=float)
        values[0, 0] = 100
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)
        target = (np.array([2.5]), np.array([2.5]))
        operator = LocalMeanInterpolator()

        # The usual 3 x 3 window excludes the corner; a square 5 x 5 includes it but a circle excludes it
        usual = raster.interp_at_points(target, method=operator, as_array=True)
        square = raster.interp_at_points(
            target,
            method=LocalMeanInterpolator(neighborhood=GridNeighbours(size=5)),
            as_array=True,
        )
        circular = raster.interp_at_points(
            target,
            method=LocalMeanInterpolator(neighborhood=GridNeighbours(size=5, shape="circular")),
            as_array=True,
        )
        assert usual[0] == 0
        assert square[0] == 4
        assert circular[0] == 0
        assert operator.default_neighborhood is None

    def test_interp_points__custom_grid_neighbours_nodata_policy(self) -> None:
        """Checks that GridNeighbours interpolation applies GDAL nodata and fixed spreading outside the method."""

        # Put nodata at the center and sample that cell plus its valid right-hand neighbor
        values = np.arange(25, dtype=np.float64).reshape(5, 5)
        values[2, 2] = np.nan
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        points = raster.ij2xy(np.array([2, 2]), np.array([2, 3]))
        operator = WindowRangeInterpolator()

        # GDAL returns nodata at the center; ignore lets the custom method use the other eight window values
        gdal_result = raster.interp_at_points(points, method=operator, as_array=True)
        ignore_result = raster.interp_at_points(
            points,
            method=operator,
            nodata_handling="ignore",
            as_array=True,
        )
        assert np.isnan(gdal_result[0])
        assert np.isfinite(gdal_result[1])
        assert np.all(np.isfinite(ignore_result))

        # A fixed one-cell spread masks both targets independently of the numerical method
        spread_result = raster.interp_at_points(
            points,
            method=operator,
            nodata_handling=1,
            as_array=True,
        )
        assert np.all(np.isnan(spread_result))

    @pytest.mark.parametrize("operator", [PropagatingLocalMeanInterpolator(), PropagatingMeanReducer()])
    def test_resample_at_points__gdal_overrides_operator_default(self, operator: Interpolator | Reducer) -> None:
        """Checks that GDAL's spatial rule omits nearby nodata despite a custom operator's propagate default."""

        # The valid center has eight nearby cells, one of which is nodata
        values = np.ones((3, 3), dtype=float)
        values[0, 0] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 3, 1, 1), crs=32631, nodata=-9999)
        point = (np.array([1.5]), np.array([1.5]))

        # The requested center is valid, so both spatial calls calculate from the eight finite cells
        result = raster.resample_at_points(point, operator, as_array=True, nodata_handling="gdal")
        np.testing.assert_array_equal(result, [1.0])

    @pytest.mark.parametrize(
        ("operator", "interpolate"),
        [(PropagatingLocalMeanInterpolator(), True), (PropagatingMeanReducer(), False)],
    )
    def test_resample_at_points__gdal_follows_custom_operator_class(
        self, operator: Interpolator | Reducer, interpolate: bool
    ) -> None:
        """Checks that a custom interpolator masks a missing source cell while a custom reducer omits it."""

        # Put one missing cell at the target and give both operators finite neighbors with value one
        values = np.ones((3, 3), dtype=float)
        values[1, 1] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 3, 1, 1), crs=32631, nodata=-9999)
        point = (np.array([1.5]), np.array([1.5]))

        # Both operators calculate one from finite cells, but only interpolation masks the center
        result = raster.resample_at_points(point, operator, as_array=True, nodata_handling="gdal")
        value = np.asarray(result).item()
        if interpolate:
            assert np.isnan(value)
        else:
            assert value == 1.0


class TestSamplingOperatorsReducers:
    """Test module for reducer selection, raster bands, window options and point outputs.

    Reducer arithmetic and fractional area references are covered in test_operators/test_reducer.py.
    """

    def test_reduce_points__reducer_object(self) -> None:
        """Checks that reduce_points() applies a Reducer to the geographic window selected by the raster."""

        # Center a three by three window on the middle value of a simple five by five raster
        values = np.arange(25, dtype=np.float64).reshape(5, 5)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)
        x, y = raster.ij2xy(np.array([2]), np.array([2]))

        # Compare the reusable Mean reducer with the existing callable form
        expected = raster.reduce_at_points((x, y), window=3, reducer_function=np.ma.mean, as_array=True)
        result = raster.reduce_at_points((x, y), window=3, reducer_function=Mean(), as_array=True)
        assert result == expected

    def test_resample_at_points__interpolator_and_reducer(self) -> None:
        """Checks that resample_at_points() samples one cell or reduces the chosen raster window."""

        # Put two nonzero values in a five by five raster so the cell value and window means differ
        values = np.zeros((5, 5), dtype=float)
        values[2, 2] = 9
        values[1, 2] = 6
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)
        x, y = raster.ij2xy(2, 2)

        # Sample the center cell, average the default 3 x 3 window, then use a five-cell circular window
        sampled = raster.resample_at_points((x, y), Nearest(neighborhood=GridNeighbours(size=1)), as_array=True)
        square_mean = raster.resample_at_points((x, y), Mean(), as_array=True)
        circular_mean = raster.resample_at_points(
            (x, y), Mean(neighborhood=GridNeighbours(size=5, shape="circular")), as_array=True
        )

        # The square has nine cells; the larger circle has 21 cell centers
        assert sampled == 9
        assert square_mean == pytest.approx(15 / 9)
        assert circular_mean == pytest.approx(15 / 21)

    def test_reduce_at_points__window_does_not_change_reducer(self) -> None:
        """Checks that a requested window does not change the reducer's own neighborhood."""

        # A nonzero center gives different means in three and five cell windows
        values = np.zeros((5, 5), dtype=float)
        values[2, 2] = 9
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)
        point = raster.ij2xy(2, 2, force_offset="center")
        window = GridNeighbours(size=5)
        reducer = Mean(neighborhood=window)

        # The window argument applies to this reduction, while resampling still uses the reducer's five cell window
        reduced = raster.reduce_at_points(point, reducer_function=reducer, window=3, as_array=True)
        resampled = raster.resample_at_points(point, reducer, as_array=True)

        # Both calculations use the center value, divided by the number of cells in their windows
        assert reduced == 1
        assert resampled == pytest.approx(9 / 25)
        assert reducer.default_neighborhood is window

    def test_resample_at_points__reducer_band_and_point_output(self) -> None:
        """Checks that a reducer reads the requested band and returns values at the requested points."""

        # Give the two bands distinct values so selecting the wrong one changes the result
        first_band = np.full((5, 5), -5, dtype=float)
        second_band = np.arange(25, dtype=float).reshape(5, 5)
        values = np.stack([first_band, second_band])
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=32631)
        points = raster.ij2xy(np.array([2, 0]), np.array([2, 0]), force_offset="center")

        # At the center, the nine values average to 12; at the corner, 0/1/5/6 average to 3
        sampled = raster.resample_at_points(points, Mean(), band=2, as_array=True)
        pointcloud = raster.resample_at_points(points, Mean(), band=2)
        np.testing.assert_array_equal(sampled, [12, 3])
        np.testing.assert_array_equal(pointcloud.data, sampled)
        np.testing.assert_array_equal(pointcloud.geometry.x, points[0])
        np.testing.assert_array_equal(pointcloud.geometry.y, points[1])
        assert pointcloud.crs == raster.crs

        # The wrapper also selects only the requested band
        reduced = raster.reduce_at_points(points, reducer_function=Mean(), window=3, band=2, as_array=True)
        np.testing.assert_array_equal(reduced, sampled)

    def test_reduce_points__reducer_window_sizes(self) -> None:
        """Checks that a Median reducer reads the requested window size and rejects invalid sizes."""

        # Put nine large values in the middle of a five by five raster and one larger center value
        values = np.zeros((5, 5), dtype=float)
        values[1:4, 1:4] = 10
        values[2, 2] = 100
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=32631)
        point = raster.ij2xy(2, 2, force_offset="center")

        # The center, 3 x 3 window, and whole raster have different middle values
        results = [
            raster.reduce_at_points(point, reducer_function=Median(), window=size, as_array=True) for size in (1, 3, 5)
        ]
        assert results == [100, 10, 0]

        # A window must have an odd, whole-number size
        with pytest.raises(ValueError, match="Window must be an odd number"):
            raster.reduce_at_points(point, reducer_function=Median(), window=2, as_array=True)
        with pytest.raises(ValueError, match="Window must be a whole number"):
            raster.reduce_at_points(point, reducer_function=Median(), window=2.5, as_array=True)  # type: ignore[arg-type]


class TestResampleOperatorNodata:
    """Test module for spatial nodata masks and contributions selected by operator neighborhoods."""

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_resample_at_points__interpolation_and_reduction_rules(self, interpolate: bool) -> None:
        """Checks that an interpolator masks a missing source cell while a reducer uses finite values."""

        # A missing center contributes to both three by three windows, but the second target is in a valid cell
        values = np.ones((5, 5), dtype=float)
        values[2, 2] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32631, nodata=-9999)
        points = (np.array([2.5, 3.25]), np.array([2.5, 2.5]))
        neighborhood = GridNeighbours(size=3)
        operator = InverseDistance(neighborhood=neighborhood) if interpolate else Mean(neighborhood=neighborhood)

        # Every finite source is one; only interpolation then masks the missing center cell
        options: dict[str, Any] = {"as_array": True}
        ignored = raster.resample_at_points(points, operator, nodata_handling="ignore", **options)
        gdal = raster.resample_at_points(points, operator, nodata_handling="gdal", **options)
        propagated = raster.resample_at_points(points, operator, nodata_handling="propagate", **options)
        np.testing.assert_allclose(ignored, [1.0, 1.0], rtol=0, atol=1e-15)
        if interpolate:
            assert np.isnan(gdal[0])
        else:
            assert gdal[0] == pytest.approx(1.0, rel=0, abs=1e-15)
        assert gdal[1] == pytest.approx(1.0, rel=0, abs=1e-15)
        assert np.all(np.isnan(propagated))


@pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
class TestKrigingRaster:
    """Test module for kriging neighborhoods, missing values and propagated uncertainty."""

    def test_krige__raster_fills_nodata_and_propagates_source_error(self) -> None:
        """Checks that Raster.krige() fills missing cells and propagates errors with its exact linear weights."""

        # Mask the center of a smooth plane while leaving a complete, symmetric set of source cells around it
        values = np.arange(9, dtype=float).reshape(3, 3)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        raster = gu.Raster.from_array(
            np.ma.masked_array(values, mask=mask),
            transform=rio.transform.from_origin(0, 3, 1, 1),
            crs=32631,
            nodata=-9999,
        )
        variogram = gu.Variogram.from_model("gaussian", effective_range=2, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Without observation errors kriging reproduces the plane; with errors it smooths noisy observations
        exact = raster.krige(variogram, max_overlap=1.5)
        nominal = raster.krige(max_overlap=1.5, variogram=variogram, error_structure=errors)
        summary = gu.uncertainty.propagate(
            raster.krige,
            error_structure=errors,
            operation_kwargs={"max_overlap": 1.5, "variogram": variogram},
        )
        np.testing.assert_allclose(exact.to_nanarray(), values, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), nominal.to_nanarray())
        assert nominal.to_nanarray()[1, 1] == pytest.approx(4)
        assert np.all(np.isfinite(np.asarray(summary.std.to_nanarray().reshape(-1))))
        assert 0 < summary.std.to_nanarray()[0, 0] < 1
        assert 0 < np.asarray(summary.std.to_nanarray().reshape(-1)).reshape(3, 3)[1, 1] < 1

    def test_interp_points__kriging_operator_uses_raster_cells(self) -> None:
        """Checks that interp_points() selects raster cells without changing the Kriging object's neighbor choice."""

        # Mask the center cell so the prediction must use nearby raster cells
        values = np.arange(9, dtype=float).reshape(3, 3)
        values[1, 1] = np.nan
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 3, 1, 1),
            crs=32631,
            nodata=-9999,
        )
        variogram = gu.Variogram.from_model("gaussian", effective_range=2, partial_sill=1)
        operator = Kriging(variogram, max_overlap=1.5)
        # Sample the center of the missing cell, rather than its upper-left corner
        x, y = np.array([1.5]), np.array([1.5])

        # Ignore the missing center and compare with the same kriging method applied to the complete raster grid
        actual = raster.interp_at_points((x, y), method=operator, as_array=True, nodata_handling="ignore")
        expected = raster.krige(variogram, max_overlap=1.5).to_nanarray()[1, 1]
        assert actual[0] == pytest.approx(expected, rel=0, abs=1e-12)
        assert operator.default_neighborhood is None

    def test_krige__shifted_raster_target_keeps_complete_physical_support(self) -> None:
        """Checks that a shifted target includes source centers entering the radius beyond its initial cell offsets."""

        # Shift one target toward the next source cells so the third center lies inside a 1.6-unit physical radius
        source = gu.Raster.from_array(
            np.array([[0.0, 0.0, 100.0]]),
            transform=rio.transform.from_origin(0, 1, 1, 1),
            crs=32631,
        )
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            transform=rio.transform.from_origin(0.49, 1, 1, 1),
            crs=32631,
        )
        variogram = gu.Variogram.from_model("gaussian", effective_range=2, partial_sill=1)

        # Compare Raster.krige() with an independent LocalData view containing all three sources in range
        coordinates = np.array([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
        target = np.array([0.99, 0.5])
        local = LocalData(
            values=np.array([0.0, 0.0, 100.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.arange(3),
            coordinates=coordinates,
            target=target,
            distances=np.linalg.norm(coordinates - target, axis=1),
        )
        expected = Kriging(variogram, max_overlap=1.6).evaluate(local)
        result = source.krige(variogram, ref=reference, max_overlap=1.6)
        assert result.to_nanarray()[0, 0] == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestInterpPointsChunked:
    """Test module for point interpolation across eager, Dask and Multiprocessing backends.

    Numerical accuracy and SciPy comparisons are covered in test_operators/test_interpolator.py.
    """

    @pytest.mark.parametrize("backend", ["eager", "dask", "multiprocessing"])
    @pytest.mark.parametrize("dtype", ["int16", "float64"])
    @pytest.mark.parametrize(
        "method,nodata_handling",
        [
            ("nearest", "gdal"),
            ("linear", "ignore"),
            ("linear", 0),
            ("linear", "propagate"),
            ("cubic", "gdal"),
            ("cubic", 0),
        ],
    )
    def test_interp_points__validity_matches_explicit_source(
        self,
        backend: str,
        dtype: str,
        method: str,
        nodata_handling: NodataChoice,
    ) -> None:
        """
        Checks that validity interpolation matches an explicit one/NaN raster with each backend and nodata rule.
        """

        # 1/ Prepare two bands with nodata in different pixels and a separate validity reference
        # Integer masks and floating NaNs must both describe unavailable pixels in the selected second band
        data = np.arange(2 * 20 * 24).reshape(2, 20, 24).astype(dtype)
        invalid = np.zeros(data.shape, dtype=bool)
        invalid[0, 3, 4] = True
        invalid[1, 7:9, 8:10] = True
        if dtype == "float64":
            data[1, 12, 14] = np.nan
        raster = gu.Raster.from_array(np.ma.array(data, mask=invalid), Affine(1, 0, 0, 0, -1, 20), 32632, nodata=-9999)

        # Construct the previous cosampling validity layer independently of the private interpolation option
        finite = np.isfinite(np.ma.getdata(raster.data[1])) & ~np.ma.getmaskarray(raster.data[1])
        validity = gu.Raster.from_array(
            np.where(finite, 1, np.nan).astype(np.float32), raster.transform, raster.crs, nodata=np.nan
        )
        rows = np.array([-2, 0, 3, 7, 8, 10, 12, 16, 19, 23])
        columns = np.array([1, 0, 4, 8, 9, 11, 14, 18, 23, 1])
        x, y = raster.ij2xy(rows, columns)
        points = (x + 0.2, y - 0.3)

        # 2/ Evaluate both representations with the same interpolation and backend settings
        # Use multiple tiles so the validity conversion must run inside the shared block kernel
        options: dict[str, Any] = {
            "points": points,
            "method": method,
            "nodata_handling": nodata_handling,
        }
        raster_input: Any = raster
        validity_input: Any = validity
        if backend == "dask":
            raster_input = raster.to_xarray().chunk({"x": 12, "y": 10}).rst
            validity_input = validity.to_xarray().chunk({"x": 12, "y": 10}).rst
        elif backend == "multiprocessing":
            options["mp_config"] = MultiprocConfig(chunks=(10, 12))
        result = raster_input.interp_at_points(band=2, as_array=True, _validity_only=True, **options)
        expected = validity_input.interp_at_points(as_array=True, **options)
        if backend == "dask":
            import dask

            result, expected = dask.compute(result, expected)

        # 3/ Check the exact finite locations and values, including points near holes and outside bounds
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, expected)
        assert np.any(np.isfinite(result))
        assert np.any(np.isnan(result))

    @pytest.mark.parametrize("validity_only", [False, True])
    def test_interp_points__validity_does_not_load_multiprocessing_source(
        self, validity_only: bool, tmp_path: Path
    ) -> None:
        """
        Checks that real workers interpolate selected values or validity while the complete raster stays unloaded.
        """

        from geoutils.multiproc.cluster import MpCluster

        # Store a raster whose nodata center is distinguishable from finite and out-of-bounds points
        data = np.arange(240, dtype=np.float64).reshape(2, 10, 12)
        data[0, 1, 1] = np.nan
        data[1, 4, 5] = np.nan
        raster = gu.Raster.from_array(data, Affine(1, 0, 0, 0, -1, 10), 32632, nodata=-9999)
        filename = tmp_path / "validity-source.tif"
        raster.to_file(filename)
        unloaded = gu.Raster(filename, load_data=False)
        points = raster.ij2xy(np.array([1, 4, 8, -2]), np.array([1, 5, 10, 1]))

        # Let worker tiles read the file and build only their local one/NaN arrays
        assert not unloaded.is_loaded
        with MpCluster({"nb_workers": 2}) as cluster:
            result = unloaded.interp_at_points(
                points,
                method="nearest",
                band=2,
                as_array=True,
                _validity_only=validity_only,
                mp_config=MultiprocConfig(chunks=(5, 6), cluster=cluster),
            )

        # Preserve the original storage and report validity at the sampled locations as floating values
        assert not unloaded.is_loaded
        assert unloaded.bands == (1, 2)
        assert result.dtype == (np.float32 if validity_only else np.float64)
        expected = [1, np.nan, 1, np.nan] if validity_only else [133, np.nan, 226, np.nan]
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("validity_only", [False, True])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_interp_points__dask_points_defer_interpolation(
        self, validity_only: bool, as_array: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Checks that sizing a lazy point result does not interpolate values or discard duplicate labels and geometry.
        """

        import dask_geopandas as dgpd

        import geoutils.interface.resampling as resampling
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # Synthetic Dask point tables bypass open_pointcloud(), which normally registers the optional accessor
        _register_dask_pointcloud_accessor()

        # Use duplicate labels and a nodata pixel so row order and optional validity conversion are both visible
        data = np.arange(30, dtype=np.float64).reshape(5, 6)
        data[2, 3] = np.nan
        raster = gu.Raster.from_array(data, Affine(1, 0, 0, 0, -1, 5), 32632, nodata=-9999)
        x, y = raster.ij2xy(np.array([1, 2, 3]), np.array([1, 3, 4]))
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), index=["a", "a", "b"], crs=raster.crs)
        lazy_points = dgpd.from_geopandas(points, npartitions=2, sort=False)

        # Reject kernel execution while constructing the graph; computing input partition lengths is allowed
        def fail_interpolation(*args: Any, **kwargs: Any) -> None:
            """Reject interpolation before the caller computes the lazy result."""

            raise AssertionError("Interpolation ran while constructing the lazy output.")

        with monkeypatch.context() as patch:
            patch.setattr(resampling, "_interp_points_base", fail_interpolation)
            output = raster.interp_at_points(
                lazy_points, method="nearest", as_array=as_array, _validity_only=validity_only
            )

        # Compute only after restoring the kernel, then compare values and the complete point output when requested
        computed = output.compute()
        expected = [1, np.nan, 1] if validity_only else [7, np.nan, 22]
        if as_array:
            np.testing.assert_array_equal(computed, expected)
        else:
            np.testing.assert_array_equal(computed["z"], expected)
            np.testing.assert_array_equal(computed.geometry.to_numpy(), points.geometry.to_numpy())
            assert computed.crs == points.crs
            np.testing.assert_array_equal(computed.index, points.index)
            assert output.pc.data_column == "z"

    def test_interp_points__boolean_outside_bounds(self) -> None:
        """Checks that nearest interpolation of booleans returns NaNs outside the raster with every backend."""

        # Alternate true and false pixels and interleave interior points with points beyond the raster
        rows, columns = np.indices((5, 6))
        values = (rows + columns) % 2 == 0
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32632)
        x, y = raster.ij2xy(np.array([1, -2, 2, 7]), np.array([1, 1, 1, 1]))
        expected = np.array([1, np.nan, 0, np.nan], dtype=np.float32)

        # Convert through integers because Raster.to_xarray() uses GDAL, which cannot store booleans
        lazy = raster.astype("uint8").to_xarray().astype(bool).chunk({"x": 3, "y": 2})

        # Interpolate the same boolean raster with eager, Dask and multiprocessing inputs
        options = {"points": (x, y), "method": "nearest", "as_array": True}
        results = [
            raster.interp_at_points(**options),
            lazy.rst.interp_at_points(**options).compute(),
            raster.interp_at_points(**options, mp_config=MultiprocConfig(chunks=(2, 3))),
        ]

        # Return exact boolean values and represent out-of-bounds samples as nodata values
        for result in results:
            assert result.dtype == np.float32
            np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("area_or_point", ["Area", "Point"])
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    @pytest.mark.parametrize("chunks", [(13, 17), (19, 11)])
    def test_interp_points__fractional_projected_coordinates_exact_backends(
        self, area_or_point: str, method: str, chunks: tuple[int, int]
    ) -> None:
        """Checks that fractional projected coordinates give exact interpolation values regardless of raster tiling."""

        import dask.array as da

        # Large coordinates and fractional pixels expose rounding from recalculating a tile's local transform
        rows, cols = np.indices((35, 43))
        values = (900 + rows * 1.3 + cols * 1.7 + 5 * np.sin(rows / 3)).astype(np.float32)
        values[10:13, 14:17] = np.nan
        transform = Affine(20, 0, 500000, 0, -20, 8600000)
        raster = gu.Raster.from_array(values, transform, 32633, nodata=-9999, area_or_point=area_or_point)
        xx, yy = raster.coords(grid=True)
        points = (xx.ravel()[::7] + 3, yy.ravel()[::7] - 4)
        options = {"points": points, "method": method, "as_array": True}

        # Interpolate the same values eagerly and with two independent rectangular block layouts
        expected = raster.interp_at_points(**options)
        lazy = raster.to_xarray().chunk({"y": chunks[0], "x": chunks[1]})
        dask_result = lazy.rst.interp_at_points(**options)
        multiproc_result = raster.interp_at_points(**options, mp_config=MultiprocConfig(chunks=(11, 15)))

        # Pixel weights and missing values must be identical for every backend
        assert isinstance(dask_result, da.Array)
        np.testing.assert_array_equal(dask_result.compute(), expected)
        np.testing.assert_array_equal(multiproc_result, expected)

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_interp_points__outer_half_pixels(self, method: Literal["nearest", "linear"]) -> None:
        """Checks that every backend keeps points on the raster's outer half pixels."""

        # Locate points on all outer half pixels and just beyond the raster
        raster = gu.Raster.from_array(np.arange(30.0).reshape(5, 6), Affine(1, 0, 0, 0, -1, 5), 32632)
        rows = np.array([-0.25, -0.25, 2, 4.25, -0.75, 4.75])
        columns = np.array([0.25, -0.25, 5.25, 5.25, 0, 0])
        x, y = raster.ij2xy(rows, columns)

        # The first four points remain inside the pixel footprint; the last two fall outside it
        expected = np.array([0 if method == "nearest" else 0.25, 0, 17, 29, np.nan, np.nan])

        # Interpolate the same points with eager, Dask and multiprocessing inputs
        lazy = raster.to_xarray().chunk({"x": 3, "y": 2})
        results = [
            raster.interp_at_points((x, y), method=method, as_array=True),
            lazy.rst.interp_at_points((x, y), method=method, as_array=True).compute(),
            raster.interp_at_points((x, y), method=method, as_array=True, mp_config=MultiprocConfig(chunks=(2, 3))),
        ]

        # Check that every backend applies the same half-pixel boundary rule
        for result in results:
            np.testing.assert_allclose(result, expected, equal_nan=True)

        # Cosampling keeps exactly the four points with finite interpolated values
        points = gu.PointCloud.from_xyz(x, y, np.ones(len(x)), crs=raster.crs)
        sample = raster.cosample(points, resample_method=method)
        np.testing.assert_array_equal(sample.ds.index, np.arange(4))
        np.testing.assert_allclose(sample.ds["self"], expected[:4])

    @pytest.mark.parametrize("path_index", [0, 2])
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    @pytest.mark.parametrize("ninterp", [2, 100])
    @pytest.mark.parametrize("as_array", [True])
    def test_interp_points__chunked_backends_equal(
        self,
        path_index: int,
        method: Literal["nearest", "linear"],
        ninterp: int,
        as_array: bool,
        lazy_test_files_tiny: list[str],
    ) -> None:
        """
        Test that interp_points yields consistent output for:
         - In-memory base function through Raster (NumPy backend),
         - In-memory base function through Xarray DataArray (NumPy backend),
         - Dask backend through Xarray accessor (lazy input and output),
         - Multiprocessing backend through Raster class (lazy input and eager output).

        Notes:
         - Dask returns one delayed array while the other backends return NumPy arrays directly,
         - Points outside of bounds are handled in the wrapper (_resample_at_points) by returning NaNs.
        """

        import dask.array as da

        # Get filepath of on-disk (for laziness) test file
        path_raster = lazy_test_files_tiny[path_index]

        # 1/ Prepare backend inputs

        # Base raster input (in-memory)
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        assert raster_base.is_loaded

        # Base data array input (in-memory)
        ds_base = open_raster(path_raster)
        ds_base.load()
        assert ds_base._in_memory

        # Multiprocessing input (lazy)
        raster_mp = gu.Raster(path_raster)
        assert not raster_mp.is_loaded

        # Dask input (lazy chunked xarray)
        ds_dask = open_raster(path_raster, chunks={"x": 10, "y": 10})
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert ds_dask.data.chunks is not None

        # 2/ Build interpolation points in georeferenced coordinates

        # We intentionally sample mostly in-bounds points (to compare actual interpolation) and a few out-of-bounds
        # points (to test the NaN padding logic)
        transform = raster_base.transform
        ny, nx = raster_base.shape  # rows, cols

        # Raster bounds in world coordinates (using rasterio for robustness)
        left, bottom, right, top = rio.transform.array_bounds(ny, nx, transform)

        rng = np.random.default_rng(seed=42)

        # Mostly in-bounds points (70%)
        nin = int(np.ceil(0.7 * ninterp))
        x_in = rng.uniform(left, right, size=nin)
        y_in = rng.uniform(bottom, top, size=nin)

        # Some out-of-bounds points (30%)
        nout = ninterp - nin
        # Push them outside by up to 20% of raster span
        dx = 0.2 * (right - left)
        dy = 0.2 * (top - bottom)
        x_out = rng.uniform(left - dx, right + dx, size=nout)
        y_out = rng.uniform(bottom - dy, top + dy, size=nout)

        x = np.concatenate([x_in, x_out])
        y = np.concatenate([y_in, y_out])

        # Shuffle so in/out are mixed (tests masking/rebuild ordering)
        perm = rng.permutation(ninterp)
        x = x[perm]
        y = y[perm]

        # 3/ Run interp_points across backends

        # The wrapper removes out-of-bounds points from chunked work and restores NaNs afterwards
        # Request arrays from every backend so output values can be compared directly
        mp_config = MultiprocConfig(chunks=(10, 7))

        # Base raster output (NumPy backend)
        out_raster = raster_base.interp_at_points(points=(x, y), method=method, as_array=as_array)

        # Base xarray output (NumPy backend)
        out_xr = ds_base.rst.interp_at_points(points=(x, y), method=method, as_array=as_array)

        # Dask output remains lazy until the comparison below
        out_dask = ds_dask.rst.interp_at_points(points=(x, y), method=method, as_array=as_array)

        # Multiprocessing output (output is eager)
        out_mp = raster_mp.interp_at_points(points=(x, y), method=method, as_array=as_array, mp_config=mp_config)

        # 4/ Laziness checks (inputs remain lazy)
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert isinstance(out_dask, da.Array)
        assert not raster_mp.is_loaded

        # 5/ Normalize to NumPy arrays
        out_raster_np = np.asarray(out_raster)
        out_xr_np = np.asarray(out_xr)
        out_dask_np = np.asarray(out_dask)
        out_mp_np = np.asarray(out_mp)

        # Shape checks
        assert out_raster_np.shape == (ninterp,)
        assert out_xr_np.shape == (ninterp,)
        assert out_dask_np.shape == (ninterp,)
        assert out_mp_np.shape == (ninterp,)

        # 6/ Compare complete outputs, including raster boundaries and out-of-bounds NaNs
        if method == "nearest":
            assert np.array_equal(out_raster_np, out_xr_np, equal_nan=True)
            assert np.array_equal(out_raster_np, out_dask_np, equal_nan=True)
            assert np.array_equal(out_raster_np, out_mp_np, equal_nan=True)
        else:
            assert np.allclose(out_raster_np, out_xr_np, equal_nan=True, rtol=1e-6, atol=0.0)
            assert np.allclose(out_raster_np, out_dask_np, equal_nan=True, rtol=1e-6, atol=0.0)
            assert np.allclose(out_raster_np, out_mp_np, equal_nan=True, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize("nodata_handling", ["gdal", "ignore", "propagate"])
    def test_interp_points__nodata_policies_backends(
        self,
        nodata_handling: NodataChoice,
        tmp_path: Path,
    ) -> None:
        """Keep each nodata policy identical across eager, Dask and Multiprocessing interpolation."""

        # Store one invalid central cell so every backend reads the same source metadata
        source = np.arange(81, dtype=np.float32).reshape(9, 9)
        source[4, 4] = np.nan
        raster = gu.Raster.from_array(
            source,
            transform=rio.transform.from_origin(0, 9, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        source_file = tmp_path / "interpolation-nodata.tif"
        raster.to_file(source_file)

        # Fractional points distinguish the three policies around the invalid cell
        x, y = raster.ij2xy(i=np.array([4.25, 3.4, 1.2]), j=np.array([4.25, 3.4, 1.2]))
        expected = raster.interp_at_points(
            (x, y),
            method="linear",
            as_array=True,
            nodata_handling=nodata_handling,
        )

        # Dask evaluates only the overlapping source chunks while preserving the selected rule
        dask_raster = open_raster(str(source_file), chunks={"x": 4, "y": 4})
        dask_output = dask_raster.rst.interp_at_points(
            (x, y),
            method="linear",
            as_array=True,
            nodata_handling=nodata_handling,
        )
        assert np.array_equal(expected, dask_output.compute(), equal_nan=True)

        # Multiprocessing applies the same rule inside independently loaded source tiles
        file_raster = gu.Raster(source_file)
        multiproc_output = file_raster.interp_at_points(
            (x, y),
            method="linear",
            as_array=True,
            nodata_handling=nodata_handling,
            mp_config=MultiprocConfig(chunks=(4, 4)),
        )
        assert np.array_equal(expected, multiproc_output, equal_nan=True)

    @pytest.mark.parametrize("raster_dask", [False, True], ids=["raster-eager", "raster-dask"])
    @pytest.mark.parametrize("point_dask", [False, True], ids=["point-eager", "point-dask"])
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_interp_points__raster_point_input_combinations(
        self,
        raster_dask: bool,
        point_dask: bool,
        method: Literal["nearest", "linear"],
        tmp_path: Path,
    ) -> None:
        """Interpolate every combination of eager and Dask raster and point-cloud inputs."""

        import dask.array as da

        # Create exact cell-center queries so both interpolation methods have one unambiguous result
        source = np.arange(36, dtype=np.int16).reshape(6, 6)
        raster = gu.Raster.from_array(
            source,
            transform=rio.transform.from_origin(0, 6, 1, 1),
            crs=32610,
            nodata=-9999,
        )
        i = np.array([1, 2, 4])
        j = np.array([1, 3, 4])
        x, y = raster.ij2xy(i=i, j=j)
        x = np.append(x, raster.bounds.right + raster.res[0])
        y = np.append(y, (raster.bounds.bottom + raster.bounds.top) / 2)
        points = gpd.GeoDataFrame(
            data={"id": np.arange(len(x))},
            geometry=gpd.points_from_xy(x=x, y=y),
            crs=raster.crs,
        )

        # Write both inputs so their Dask variants read the same values and metadata
        raster_file = tmp_path / "raster.tif"
        point_file = tmp_path / "points.gpkg"
        raster.to_file(raster_file)
        points.to_file(point_file)
        raster_input = open_raster(str(raster_file), chunks={"x": 3, "y": 3}) if raster_dask else raster
        point_input = gu.open_pointcloud(str(point_file), data_column="id", chunks=2) if point_dask else points

        # The complete in-memory calculation defines the exact expected values
        expected = raster.interp_at_points(points=points, method=method, as_array=True)
        output = (
            raster_input.rst.interp_at_points(points=point_input, method=method, as_array=True)
            if raster_dask
            else raster_input.interp_at_points(points=point_input, method=method, as_array=True)
        )

        # Either Dask input must produce a lazy array while two eager inputs return NumPy directly
        if raster_dask or point_dask:
            assert isinstance(output, da.Array)
            computed_output = output.compute()
        else:
            assert isinstance(output, np.ndarray)
            computed_output = output
        assert np.array_equal(expected, computed_output, equal_nan=True)

        # Evaluating the result must leave each Dask input unchanged and lazy
        if raster_dask:
            assert not raster_input._in_memory
            assert isinstance(raster_input.data, da.Array)
        if point_dask:
            assert not point_input.pc.is_loaded

    def test_interp_points__dask_pointcloud_multiprocessing_error(self, tmp_path: Path) -> None:
        """Reject Multiprocessing when Dask already partitions the point-cloud input."""

        # Store one point source and reopen it lazily through the public helper
        raster = gu.Raster.from_array(
            np.arange(9, dtype=np.float32).reshape(3, 3),
            transform=rio.transform.from_origin(0, 3, 1, 1),
            crs=32610,
        )
        x, y = raster.ij2xy(i=np.array([1]), j=np.array([1]))
        points = gpd.GeoDataFrame(
            data={"id": [1]},
            geometry=gpd.points_from_xy(x=x, y=y),
            crs=raster.crs,
        )
        point_file = tmp_path / "points.gpkg"
        points.to_file(point_file)
        dask_points = gu.open_pointcloud(str(point_file), data_column="id", chunks=1)

        # One operation cannot be scheduled by both Dask and Multiprocessing
        with pytest.raises(ValueError, match="Dask point-cloud inputs cannot be combined with Multiprocessing"):
            raster.interp_at_points(
                points=dask_points,
                method="nearest",
                as_array=True,
                mp_config=MultiprocConfig(chunks=(2, 2)),
            )
        assert not dask_points.pc.is_loaded

    def test_interp_points__dask_pointcloud_input(self, lazy_test_files_tiny: list[str]) -> None:
        """Test interpolation to Dask-GeoPandas point-cloud inputs."""

        # Load the lazy dataframe and array types used in assertions
        import dask.array as da
        import dask_geopandas as dgpd

        # Compare a loaded Raster with a chunked accessor over the same source
        path_raster = lazy_test_files_tiny[0]
        raster = gu.Raster(path_raster)
        raster.load()
        ds_dask = open_raster(path_raster, chunks={"x": 10, "y": 10})

        # Include two raster-edge points and one truly out-of-bounds point
        left, bottom, right, top = rio.transform.array_bounds(*raster.shape, raster.transform)
        x = np.array([left, (left + right) / 2, right + raster.res[0]])
        y = np.array([top, (bottom + top) / 2, bottom - raster.res[1]])
        points = gpd.GeoDataFrame({"id": [1, 2, 3]}, geometry=gpd.points_from_xy(x, y), crs=raster.crs)

        # Open the points in two lazy partitions through the public helper
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "points.gpkg")
        points.to_file(temp_file)
        dask_points = gu.open_pointcloud(temp_file, data_column="id", chunks=2)

        # Compute the eager values once as the expected result
        expected = raster.interp_at_points(points=points, method="nearest", as_array=True)

        # Array output should remain lazy and preserve out-of-bounds NaNs
        out_array = ds_dask.rst.interp_at_points(points=dask_points, method="nearest", as_array=True)
        assert isinstance(out_array, da.Array)
        assert np.array_equal(np.asarray(expected), out_array.compute(), equal_nan=True)
        assert not ds_dask._in_memory
        assert not dask_points.pc.is_loaded

        # Point-cloud output should remain a Dask-GeoPandas collection with ``pc`` metadata
        out_points = ds_dask.rst.interp_at_points(points=dask_points, method="nearest")
        assert isinstance(out_points, dgpd.GeoDataFrame)
        assert not out_points.pc.is_loaded
        assert np.array_equal(np.asarray(expected), out_points.compute()["z"].to_numpy(), equal_nan=True)
        assert not ds_dask._in_memory
        assert not dask_points.pc.is_loaded
        assert not out_points.pc.is_loaded

    def test_interp_points_las__dask_pointcloud_input(self) -> None:
        """Test interpolation to Dask point-cloud inputs opened from LAS."""

        # The class marker covers lazy GeoDataFrames; this case also needs the optional LAS reader
        pytest.importorskip("laspy")
        import dask.array as da

        # Surround the LAS extent with a simple constant raster
        fn_las = gu.examples.get_path_test("coromandel_lidar")
        pc = gu.PointCloud(fn_las)
        bounds = pc.bounds
        xpad = bounds.right - bounds.left
        ypad = bounds.top - bounds.bottom
        transform = rio.transform.from_bounds(
            bounds.left - xpad,
            bounds.bottom - ypad,
            bounds.right + xpad,
            bounds.top + ypad,
            width=4,
            height=4,
        )
        raster = gu.Raster.from_array(data=np.full((4, 4), 5, dtype=np.float32), transform=transform, crs=pc.crs)

        # Interpolation should compute LAS partitions without loading the accessor
        dask_points = gu.open_pointcloud(fn_las, chunks=100)
        output = raster.interp_at_points(points=dask_points, method="nearest", as_array=True)

        # Every in-bounds point must receive the constant raster value
        assert isinstance(output, da.Array)
        assert np.array_equal(output.compute(), np.full(dask_points.pc.point_count, 5, dtype=np.float32))
        assert not dask_points.pc.is_loaded


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestSamplingOperatorsChunked:
    """Test module for eager, Dask and multiprocessing agreement when sampling raster neighborhoods."""

    def test_interp_points__custom_grid_neighbours_chunk_invariance(self) -> None:
        """Checks that selected source cells and values stay exact across eager, Dask and multiprocessing tiles."""

        # Place missing values across rectangular chunk boundaries and sample cell centers on either side
        values = np.arange(72, dtype=np.float64).reshape(8, 9)
        values[2, 3] = np.nan
        values[5, 6] = np.nan
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 8, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        rows = np.array([1, 2, 3, 4, 5, 6])
        cols = np.array([2, 3, 4, 5, 6, 7])
        points = raster.ij2xy(rows, cols)
        options = {"points": points, "method": WindowRangeInterpolator(), "as_array": True}

        # Evaluate the same source windows from the complete array and two independent chunk layouts
        expected = raster.interp_at_points(**options)
        lazy = raster.to_xarray().chunk({"y": 3, "x": 4})
        dask_result = lazy.rst.interp_at_points(**options).compute()
        multiproc_result = raster.interp_at_points(**options, mp_config=MultiprocConfig(chunks=(4, 3)))

        # Check that reading neighboring chunks selects the same source cells and gives the same values
        np.testing.assert_array_equal(dask_result, expected)
        np.testing.assert_array_equal(multiproc_result, expected)

    def test_interp_points__automatic_grid_neighbours_chunk_invariance(self) -> None:
        """Checks that automatic raster windows give the same values across eager and chunked calculations."""

        # Split an eight by nine raster into 3 x 4 cells, with targets on both sides of chunk edges
        values = np.arange(72, dtype=float).reshape(8, 9)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 8, 1, 1), crs=4326)
        points = (np.array([2.5, 3.5, 5.5]), np.array([5.5, 4.5, 2.5]))
        operator = LocalMeanInterpolator()

        # Build a lazy output before computing it, then run the same method in worker tiles of 4 x 3 cells
        expected = raster.interp_at_points(points, method=operator, as_array=True)
        lazy_source = raster.to_xarray().chunk({"y": 3, "x": 4})
        lazy_result = lazy_source.rst.interp_at_points(points, method=operator, as_array=True)
        assert hasattr(lazy_source.data, "compute")
        assert hasattr(lazy_result, "compute")
        multiproc_result = raster.interp_at_points(
            points,
            method=operator,
            as_array=True,
            mp_config=MultiprocConfig(chunks=(4, 3)),
        )

        # Each backend must use the same source window, including cells across chunk boundaries
        np.testing.assert_array_equal(lazy_result.compute(), expected)
        np.testing.assert_array_equal(multiproc_result, expected)


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestRegularInterpolationNeighboursChunked:
    """Test module for lazy raster sampling with explicit natural stencils at chunk boundaries."""

    @pytest.mark.parametrize("method", ["nearest", "linear", "slinear", "pchip"])
    def test_resample_at_points__natural_stencil_chunk_invariance(self, method: ScipyInterpolationMethod) -> None:
        """Checks that explicit natural stencils remain lazy and match eager values across raster chunk boundaries."""

        # Uneven 5 x 6 chunks split a nonlinear surface near several requested points
        rows, cols = np.indices((13, 14), dtype=float)
        raster = gu.Raster.from_array(rows * cols + np.sin(cols), rio.transform.from_origin(0, 13, 1, 1), crs=32631)
        source = raster.to_xarray().chunk({"y": 5, "x": 6})
        axis = (0,) if method == "nearest" else (-1, 0, 1, 2) if method == "pchip" else (0, 1)
        operator = ScipyInterpolator(
            method,
            neighborhood=GridNeighbours(tuple((row, col) for row in axis for col in axis)),
        )
        targets = (np.array([5.8, 6.2, 10.1]), np.array([8.1, 7.8, 3.1]))

        # Building the result must leave both the source and the output backed by Dask
        expected = raster.resample_at_points(targets, operator, as_array=True)
        result = source.rst.resample_at_points(targets, operator, as_array=True)
        assert hasattr(source.data, "compute")
        assert hasattr(result, "compute")
        computed = result.compute()
        if method == "slinear":
            # Separate spline fits change the rounding of coefficient reductions by a few floating-point units
            np.testing.assert_allclose(computed, expected, rtol=0, atol=5e-14)
        else:
            np.testing.assert_array_equal(computed, expected)


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
class TestReductionChunked:
    """Test module for reducer windows, callable reductions and lazy point output across Dask/MP chunks.

    Window accuracy and fractional area calculations are covered in test_operators/test_reducer.py.
    """

    @pytest.mark.parametrize("window,shape", [(1, "square"), (5, "square"), (11, "square"), (5, "circular")])
    @pytest.mark.parametrize("fractional", [False, True])
    def test_resample_at_points__reducer_chunk_overlap(
        self, window: int, shape: Literal["square", "circular"], fractional: bool, tmp_path: Path
    ) -> None:
        """Checks that overlapping reducer windows give the eager result across Dask/MP tiles and raster edges."""

        # Distinct bands, missing cells near chunk boundaries, and windows larger than some raster axes
        values = np.arange(63, dtype=float).reshape(7, 9) ** 2
        values[2, 3] = np.nan
        values[5, 7] = np.nan
        raster = gu.Raster.from_array(
            np.stack([np.full_like(values, -10), values]), Affine(1, 0, 0, 0, -1, 7), 32631, nodata=-9999
        )
        filename = tmp_path / "reduction.tif"
        raster.to_file(filename)
        points = (np.array([8.75, 0.25, 4.25, 2.5, 10]), np.array([0.25, 6.75, 4.25, 3.5, 4]))
        operator = Sum(neighborhood=GridNeighbours(size=window, shape=shape))
        options: dict[str, Any] = {
            "method": operator,
            "coverage": ("fractional" if fractional else "center"),
            "band": 2,
            "as_array": True,
        }

        # Uneven Dask/MP tiles force overlap reads and a shorter final chunk
        expected = raster.resample_at_points(points, **options)
        lazy_source = open_raster(str(filename), chunks={"y": 2, "x": 4})
        unloaded = gu.Raster(filename, load_data=False)
        lazy_result = lazy_source.rst.resample_at_points(points, **options)
        mp_result = unloaded.resample_at_points(points, **options, mp_config=MultiprocConfig(chunks=(3, 2)))

        # Only worker tiles loaded; lazy values and source stay unloaded until compute()
        assert not lazy_source._in_memory
        assert hasattr(lazy_result, "compute")
        assert not unloaded.is_loaded
        assert unloaded.bands == (1, 2)
        assert isinstance(mp_result, np.ndarray)
        np.testing.assert_array_equal(lazy_result.compute(), expected)
        np.testing.assert_array_equal(mp_result, expected)

    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("boundless", [False, True])
    def test_reduce_at_points__callable_workers(self, masked: bool, boundless: bool, tmp_path: Path) -> None:
        """Checks that callable reducers receive missing cells consistently in eager, Dask and MP windows."""

        from geoutils.multiproc.cluster import MpCluster

        # Missing cell shared by neighboring windows; median differs from the containing cell
        values = np.arange(63, dtype=float).reshape(7, 9) ** 2
        values[2, 3] = np.nan
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 7), 32631, nodata=-9999)
        filename = tmp_path / "callable-reduction.tif"
        raster.to_file(filename)
        points = raster.ij2xy(np.array([2, 3, 0, 6]), np.array([3, 4, 0, 8]), force_offset="center")
        function = np.ma.median if masked else np.nanmedian
        options: dict[str, Any] = {
            "reducer_function": function,
            "window": 5,
            "masked": masked,
            "as_array": True,
            "boundless": boundless,
        }

        # Shared dispatcher with 2 x 4 Dask chunks and real MP workers reading 3 x 2 tiles
        expected = raster.reduce_at_points(points, **options)
        lazy_source = open_raster(str(filename), chunks={"y": 2, "x": 4})
        unloaded = gu.Raster(filename, load_data=False)
        lazy_result = lazy_source.rst.reduce_at_points(points, **options)
        with MpCluster({"nb_workers": 2}) as cluster:
            mp_result = unloaded.reduce_at_points(
                points, **options, mp_config=MultiprocConfig(chunks=(3, 2), cluster=cluster)
            )

        # Eager values, masks and original loading state
        assert not lazy_source._in_memory
        assert hasattr(lazy_result, "compute")
        assert not unloaded.is_loaded
        assert isinstance(mp_result, np.ndarray)
        np.testing.assert_array_equal(lazy_result.compute(), expected)
        np.testing.assert_array_equal(mp_result, expected)

    @pytest.mark.parametrize("lazy_points", [False, True])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_reduce_at_points__lazy_output(
        self, lazy_points: bool, as_array: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that constructing a Dask reduction result defers calculation for arrays and point clouds."""

        import geoutils.interface.resampling as resampling
        from geoutils._misc import import_optional

        # Boundary points and an outside target; duplicate labels in partitioned point input
        values = np.arange(35, dtype=float).reshape(5, 7)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        x, y = np.array([0.25, 2.5, 6.75, 8]), np.array([4.75, 2.5, 0.25, 1])
        points = gu.PointCloud.from_xyz(x, y, np.zeros(4), crs=raster.crs).ds
        points.index = [2, 2, 7, 9]
        source = raster.to_xarray().chunk({"y": 2, "x": 3})
        expected = raster.reduce_at_points((x, y), reducer_function=Sum(), window=3, as_array=True)
        if lazy_points:
            dask_geopandas = import_optional("dask_geopandas")
            targets = dask_geopandas.from_geopandas(points, npartitions=2, sort=False)
        else:
            targets = (x, y)

        # Building the graph can read point counts, but must not run the reduction
        def fail_reduction(*args: Any, **kwargs: Any) -> Any:
            """Reject calculation before compute()."""

            raise AssertionError("Reduction ran before compute()")

        with monkeypatch.context() as patch:
            patch.setattr(resampling, "_resample_points_base", fail_reduction)
            result = source.rst.reduce_at_points(targets, reducer_function=Sum(), window=3, as_array=as_array)
        assert not source._in_memory
        assert hasattr(result, "compute")

        # Computed values match the eager call; point output preserves target geometry and labels
        computed = result.compute()
        np.testing.assert_array_equal(computed if as_array else computed["z"], expected)
        if not as_array:
            np.testing.assert_array_equal(computed.geometry.x, x)
            np.testing.assert_array_equal(computed.geometry.y, y)
            if lazy_points:
                np.testing.assert_array_equal(computed.index, points.index)


@pytest.mark.skipif(find_spec("dask") is None, reason="Requires Dask")
@pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
class TestKrigingRasterChunked:
    """Test module for lazy and multiprocessing kriging across chunk boundaries."""

    def test_interp_points__kriging_operator_chunk_invariance(self) -> None:
        """Checks that interp_points() reads the same kriging neighbors eagerly and across raster chunks."""

        # Put missing cells beside uneven chunk boundaries and sample targets on both sides
        values = np.arange(42, dtype=float).reshape(6, 7)
        values[2, 3] = np.nan
        values[4, 5] = np.nan
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 6, 1, 1),
            crs=32631,
            nodata=-9999,
        )
        points = (np.array([2.5, 3.5, 5.5]), np.array([3.5, 3.5, 1.5]))
        variogram = gu.Variogram.from_model("gaussian", effective_range=3, partial_sill=1)
        operator = Kriging(variogram, max_overlap=2.2)

        # Compare the eager values with a lazy raster split into 2 x 3 cells and worker tiles of 3 x 4 cells
        expected = raster.interp_at_points(points, method=operator, as_array=True, nodata_handling="ignore")
        lazy_source = raster.to_xarray().chunk({"y": 2, "x": 3})
        lazy_result = lazy_source.rst.interp_at_points(points, method=operator, as_array=True, nodata_handling="ignore")
        assert hasattr(lazy_source.data, "compute")
        assert hasattr(lazy_result, "compute")
        multiproc_result = raster.interp_at_points(
            points,
            method=operator,
            as_array=True,
            nodata_handling="ignore",
            mp_config=MultiprocConfig(chunks=(3, 4)),
        )

        # Both chunk layouts give the eager predictions
        np.testing.assert_allclose(lazy_result.compute(), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(multiproc_result, expected, rtol=0, atol=1e-12)

    def test_krige__raster_chunk_invariance(self, tmp_path: Path) -> None:
        """Checks that raster kriging uses all nearby source cells across Dask and multiprocessing chunks."""

        # Put missing cells beside uneven chunk boundaries so their source windows need neighboring blocks
        values = np.arange(42, dtype=float).reshape(6, 7)
        values[2, 3] = np.nan
        values[4, 5] = np.nan
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 6, 1, 1),
            crs=32631,
            nodata=-9999,
        )
        variogram = gu.Variogram.from_model("gaussian", effective_range=3, partial_sill=1)

        # Use the same search radius eagerly, with Dask chunks of 2 x 3 cells, and with worker tiles of 3 x 4 cells
        expected = raster.krige(variogram, max_overlap=2.2)
        lazy_source = raster.to_xarray().chunk({"y": 2, "x": 3})
        lazy = lazy_source.rst.krige(variogram, max_overlap=2.2)
        multiproc = raster.krige(
            variogram,
            max_overlap=2.2,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(tmp_path / "kriging.tif")),
        )

        # Lazy input and output remain unloaded until compute(), and both chunked results equal the eager raster
        assert not lazy_source._in_memory
        assert hasattr(lazy.data, "compute")
        np.testing.assert_allclose(np.asarray(lazy.compute()).squeeze(), expected.to_nanarray(), rtol=0, atol=1e-12)
        assert expected.raster_equal(multiproc, strict_masked=False)


class TestResamplingEdgeCases:
    """Test module for partial windows, missing or outside targets and invalid resampling options."""

    def test_resample_at_points__reducer_edge_and_outside(self) -> None:
        """Checks that edge windows use the cells inside the raster and outside points return NaN."""

        # Select an interior cell, two corners, and a point just below the raster
        values = np.arange(1, 17, dtype=float).reshape(4, 4)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 4, 1, 1), crs=32631)
        rows = np.array([1, 0, 3, 4])
        cols = np.array([2, 0, 3, 1])
        points = raster.ij2xy(rows, cols, force_offset="center")

        # The first 3 x 3 window has nine cells; each corner window has only four
        result = raster.resample_at_points(points, Sum(), as_array=True)
        np.testing.assert_array_equal(result[:3], [63, 14, 54])
        assert np.isnan(result[3])

        # The older public name uses the same reducer path when given a window
        reduced = raster.reduce_at_points(points, reducer_function=Sum(), window=3, as_array=True)
        np.testing.assert_allclose(reduced, result, rtol=0, atol=0, equal_nan=True)

    @pytest.mark.parametrize("fractional", [False, True])
    @pytest.mark.parametrize("boundless", [False, True])
    def test_reduce_at_points__partial_windows(self, fractional: bool, boundless: bool) -> None:
        """Checks that edge windows use covered cells or return NaN when complete windows are required."""

        # Constant raster: four cells at the corner, nine around the interior point
        raster = gu.Raster.from_array(np.ones((4, 4)), Affine(1, 0, 0, 0, -1, 4), 32631)
        points = (np.array([0.5, 1.5]), np.array([3.5, 2.5]))
        result = raster.reduce_at_points(
            points,
            reducer_function=Sum(),
            window=3,
            coverage=("fractional" if fractional else "center"),
            boundless=boundless,
            as_array=True,
        )

        # Both ordinary and fractional windows have the same area at cell centers
        np.testing.assert_array_equal(result, [4 if boundless else np.nan, 9])

    def test_reduce_at_points__circular_window_boundless(self) -> None:
        """Checks that a size-five circle needs its full 2.5-pixel radius inside the raster."""

        # The first circle crosses the left edge; the second touches it without crossing
        raster = gu.Raster.from_array(np.ones((7, 7)), Affine(1, 0, 0, 0, -1, 7), 32631)
        points = (np.array([2.25, 2.5]), np.array([3.5, 3.5]))

        # Require the full circular area for both requested points
        result = raster.reduce_at_points(
            points, window=5, window_shape="circular", coverage="fractional", boundless=False, as_array=True
        )
        assert np.isnan(result[0])
        assert result[1] == 1

    @pytest.mark.parametrize("method", [Mean(), np.nanmean])
    @pytest.mark.parametrize("fractional,masked", [(False, False), (False, True), (True, False)])
    def test_resample_at_points__window_options(self, method: Any, fractional: bool, masked: bool) -> None:
        """Checks that shared resampling and reduction options give the same window mean and missing targets."""

        # Missing neighbor and a high center value, giving a window mean different from the center
        values = np.arange(25, dtype=float).reshape(5, 5)
        values[1, 1] = np.nan
        values[2, 2] = 100
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32606, nodata=-99999)
        points = raster.ij2xy(np.array([0, 2, 9]), np.array([0, 2, 9]), force_offset="center")

        # Full windows only: one edge target, one complete window and one outside target
        options: dict[str, Any] = {
            "window": 3,
            "coverage": ("fractional" if fractional else "center"),
            "masked": masked,
            "boundless": False,
            "as_array": True,
        }
        resampled = raster.resample_at_points(points, method=method, **options)
        reduced = raster.reduce_at_points(points, reducer_function=method, **options)

        # Eight finite neighbors sum to 190; missing targets become NaN or masked values
        expected = np.array([np.nan, 190 / 8, np.nan])
        for result in (resampled, reduced):
            np.testing.assert_array_equal(np.ma.filled(result, np.nan), expected)
            if masked:
                np.testing.assert_array_equal(np.ma.getmaskarray(result), [True, False, True])

    def test_resample_at_points__reducer_nodata_distance(self) -> None:
        """Checks that a reducer omits nodata under GDAL and masks nearby cells with a distance choice."""

        # Make the center cell missing and query it together with the cell to its right
        values = np.ones((5, 5), dtype=float)
        values[2, 2] = np.nan
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 5, 1, 1), crs=32631, nodata=-9999)
        points = raster.ij2xy(np.array([2, 2]), np.array([2, 3]), force_offset="center")

        # Omitting the missing cell leaves two means of one; a distance choice masks both targets
        ignored = raster.resample_at_points(points, Mean(), as_array=True, nodata_handling="ignore")
        gdal = raster.resample_at_points(points, Mean(), as_array=True, nodata_handling="gdal")
        spread = raster.resample_at_points(points, Mean(), as_array=True, nodata_handling=1)
        np.testing.assert_array_equal(ignored, [1, 1])
        np.testing.assert_array_equal(gdal, [1, 1])
        assert np.all(np.isnan(spread))

    @pytest.mark.parametrize("operator", [Mean(), Nearest()])
    def test_resample_at_points__error_neighborhood_argument(self, operator: Reducer | Interpolator) -> None:
        """Checks that a neighborhood is configured on the operator, not on resample_at_points()."""

        # Only the call signature matters here; the point is inside a complete raster window
        raster = gu.Raster.from_array(np.ones((3, 3)), rio.transform.from_origin(0, 3, 1, 1), crs=4326)
        point = raster.ij2xy(1, 1, force_offset="center")

        # Passing the neighborhood to the method gives a clear instruction for either operator type
        with pytest.raises(TypeError, match="Set the neighborhood on the Interpolator or Reducer"):
            raster.resample_at_points(point, operator, neighborhood=GridNeighbours(size=3))

    def test_resample_at_points__error_point_neighborhood_for_reducer(self) -> None:
        """Checks that raster reduction rejects a neighborhood intended for point cloud observations."""

        # The raster has a complete neighborhood around the selected center cell
        raster = gu.Raster.from_array(np.ones((3, 3)), rio.transform.from_origin(0, 3, 1, 1), crs=4326)
        point = raster.ij2xy(1, 1, force_offset="center")
        reducer = Mean(neighborhood=PointNeighbours(k=2))

        # PointNeighbours cannot name raster cell offsets
        with pytest.raises(ValueError, match="PointNeighbours applies to point sources"):
            raster.resample_at_points(point, reducer)

    def test_reduce_points__fractional_area_requires_weighted_reducer(self) -> None:
        """Checks that fractional reduction rejects methods that cannot use covered cell areas."""

        # This custom cell selection has no square/circle footprint, and the methods do not accept area weights
        raster = gu.Raster.from_array(np.ones((3, 3)), rio.transform.from_origin(0, 3, 1, 1), crs=32631)
        point = (1.5, 1.5)

        # A custom reducer must declare support for geometric weights
        with pytest.raises(ValueError, match="does not accept support_weights"):
            raster.reduce_at_points(
                point, reducer_function=NoSupportReducer(), window=1, coverage="fractional", as_array=True
            )
        with pytest.raises(TypeError, match="requires a Reducer"):
            raster.reduce_at_points(
                point, reducer_function=np.ma.median, window=1, coverage="fractional", as_array=True
            )
        with pytest.raises(ValueError, match="square or circular GridNeighbours window"):
            raster.resample_at_points(
                point, Mean(neighborhood=GridNeighbours(((0, 0), (0, 1)))), coverage="fractional", as_array=True
            )

    @pytest.mark.parametrize("method", ["reduce_at_points", "reduce_points"])
    def test_reduce_at_points__error_return_window(self, method: str) -> None:
        """Checks that the reduction API rejects the removed return_window argument."""

        raster = gu.Raster.from_array(np.ones((3, 3)), Affine(1, 0, 0, 0, -1, 3), 32631)
        with pytest.raises(TypeError, match="return_window"):
            getattr(raster, method)((1.5, 1.5), return_window=True)
