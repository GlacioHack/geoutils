"""Test interpolator classes, both on regular and irregular inputs."""

from __future__ import annotations

import subprocess
import warnings
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from affine import Affine
from scipy.interpolate import interpn
from scipy.ndimage import binary_dilation, map_coordinates

import geoutils as gu
from benchmarks.comparisons.gdal import build_gdal_grid_command
from geoutils import Raster
from geoutils._misc import import_optional, silence_rasterio_message
from geoutils._typing import NDArrayNum
from geoutils.interface.gridding import GriddingMethod, _grid_pointcloud
from geoutils.interface.resampling import _interpolate_array, _resample_at_points
from geoutils.operators import GridNeighbours, Interpolator, LinearCoefficients, LocalData, PointNeighbours
from geoutils.operators.interpolator import INTERPOLATION_ORDERS as method_to_order
from geoutils.operators.interpolator import (
    Cubic,
    InverseDistance,
    Kriging,
    Linear,
    Nearest,
    ScipyInterpolationMethod,
    ScipyInterpolator,
    _get_dist_nodata_spread,
    _interpn_interpolator,
)
from geoutils.operators.nodata import NodataChoice
from geoutils.stats.variography import VariogramModel
from tests.operator_helpers import LocalMeanInterpolator


# Custom subclass implementing only coefficients() for tests
class FixedAffineInterpolator(Interpolator):
    def coefficients(self, data: LocalData) -> LinearCoefficients:
        return LinearCoefficients(weights=np.array([1.0, 0.0]), offset=2.0)


class TestInterpolator:
    """
    Test module for the Interpolator class.

    We test the use of linear coefficients for weights/covariance, behaviour with nodata and
    subclass overrides.

    Tests specific to regular/irregular inputs are further below in TestInterpolatorRegular/Irregular.
    Tests specific to gridding/resampling at points and chunked implementations with Dask/MP are located in
    test_interface/test_resampling.py and test_interface/test_gridding.py.
    """

    def test_interpolator__linearcoef(self) -> None:
        """Checks that coefficients calculate the exact weighted value and constant defined by the method."""

        # Evaluate coefficients
        data = LocalData(values=np.array([3.0, 8.0]), valid=np.ones(2, dtype=bool), source_ids=np.arange(2))
        result = FixedAffineInterpolator().evaluate(data)

        # Check linear result
        assert result == 5.0

    def test_interpolator__predict(self) -> None:
        """Checks that predict() receives finite observations and returns batches in input order."""

        # Create partially and entirely missing groups
        data = LocalData(values=np.array([2.0, np.nan, 6.0]), valid=np.ones(3, dtype=bool), source_ids=np.arange(3))
        missing = LocalData(values=np.array([np.nan]), valid=np.ones(1, dtype=bool), source_ids=np.array([4]))
        operator = LocalMeanInterpolator()

        # Check order and nodata handling against evaluate()
        result = operator.predict_batch([missing, data])
        np.testing.assert_array_equal(result, [np.nan, 4.0])
        assert operator.evaluate(data) == 4.0
        assert np.isnan(operator.evaluate(data, nodata_propagation="propagate"))

    @pytest.mark.parametrize("invalid_index,expected", [(0, np.nan), (1, 5.0)])
    def test_interpolator__validity(self, invalid_index: int, expected: float) -> None:
        """Checks that an affine result becomes invalid for a non-zero coef applied to an invalid value."""

        valid = [True, True]
        valid[invalid_index] = False
        values = [3.0, 8.0]
        values[invalid_index] = np.nan
        data = LocalData(values=np.asarray(values), valid=np.asarray(valid), source_ids=np.arange(2))

        # Check nodata propagation
        result = FixedAffineInterpolator().evaluate(data, nodata_propagation="propagate")
        assert np.array_equal(result, expected, equal_nan=True)

    @pytest.mark.parametrize("method", ["cubic", "slinear"])
    def test_interpolator__scipy_subclass_overrides_prediction(self, method: ScipyInterpolationMethod) -> None:
        """Checks that a SciPy subclass with its own prediction can use grid or point neighborhoods."""

        class RangeInterpolator(ScipyInterpolator):
            """Replace SciPy interpolation with the range of the selected observations."""

            def predict(self, data: LocalData) -> float:
                """Return the difference between the largest and smallest selected values."""

                return float(np.ptp(data.values))

        # Raster and point neighborhoods for overridden predict()
        raster = gu.Raster.from_array(np.arange(25.0).reshape(5, 5), rio.transform.from_origin(0, 5, 1, 1), crs=32631)
        points = gu.PointCloud.from_xyz([0, 1, 10], [0, 0, 0], [2, 4, 100], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 0, 1, 1), crs=32631)
        raster_operator = RangeInterpolator(method, neighborhood=GridNeighbours(size=3))
        point_operator = RangeInterpolator(method, neighborhood=PointNeighbours(k=2))

        # Check ranges (6 to 18 for raster window, 2 to 4 for nearest points)
        sampled = raster.resample_at_points((2.5, 2.5), raster_operator, as_array=True)
        gridded = points.grid(ref=reference, resampling=point_operator)
        assert sampled[0] == 12
        assert gridded.to_nanarray()[0, 0] == 2

    def test_interpolator__error_support_weights(self) -> None:
        """Checks that a custom interpolator cannot enable support weights (i.e. fractional area)."""

        class ClaimsAreaSupport(FixedAffineInterpolator):
            accepts_support_weights = True

        # Should not be accepted
        data = LocalData(
            values=np.array([3.0, 8.0]),
            valid=np.ones(2, dtype=bool),
            source_ids=np.arange(2),
            support_weights=np.array([0.25, 0.75]),
        )
        with pytest.raises(ValueError, match="does not accept support_weights"):
            ClaimsAreaSupport().evaluate(data)


class TestInterpolatorRegular:
    """
    Test module for regular interpolation.

    This includes testing gridded interpolation optimizations, grid neighborhoods and incompatible windows.
    """

    @pytest.mark.parametrize("operator,expected", [(Nearest(), 2.0), (Linear(), 6.0)])
    def test_resample_at_points__grid(self, operator: Interpolator, expected: float) -> None:
        """Checks that nearest selects one value and bilinear the four surrounding ones."""

        # Nonplanar surface (to distinguish bilinear from triangle interpolation)
        values = np.array([[2.0, 6.0], [10.0, 30.0]])
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), crs=32631)
        target = raster.ij2xy(0.25, 0.25)

        # Row/column offsets of 0.25 give weights (3/4, 1/4) along each axis
        # Bilinear weights are their products: [[9/16, 3/16], [3/16, 1/16]]
        # Expected value for bilinear: (2*9 + 6*3 + 10*3 + 30*1) / 16 = 6
        # Nearest: 2
        result = raster.resample_at_points(target, operator, as_array=True)
        np.testing.assert_array_equal(result, [expected])

    @pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"])  # type: ignore
    def test_interpn_interpolator__matches_scipy(self, method: ScipyInterpolationMethod) -> None:
        """Checks that regular interpolation returns exactly the same result as scipy.interpolate.interpn()."""

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

    @pytest.mark.parametrize("method", ["nearest", "linear", "slinear", "pchip"])
    def test_resample_at_points__exact_stencil(self, method: ScipyInterpolationMethod) -> None:
        """Checks that the exact natural stencil is accepted without warnings and leaves interpolation unchanged."""

        # Direct offsets for even-width linear and PCHIP stencils
        axis = (0,) if method == "nearest" else (-1, 0, 1, 2) if method == "pchip" else (0, 1)
        neighborhood = GridNeighbours(tuple((row, col) for row in axis for col in axis))
        values = np.arange(81, dtype=float).reshape(9, 9) ** 2
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 9, 1, 1), crs=32631)
        targets = (np.array([2.3, 5.1]), np.array([6.6, 3.7]))

        # Treat warnings as errors for exact stencil
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            result = raster.resample_at_points(
                targets,
                ScipyInterpolator(method, neighborhood=neighborhood),
                as_array=True,
            )
        expected = raster.interp_at_points(targets, method=method, as_array=True)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("method,size", [("nearest", 3), ("linear", 3), ("slinear", 3), ("pchip", 5)])
    @pytest.mark.parametrize("nodata_handling", ["ignore", "gdal", "propagate"])
    def test_resample_at_points__extra_offsets_preserve_regular_method(
        self, method: ScipyInterpolationMethod, size: int, nodata_handling: NodataChoice
    ) -> None:
        """Checks that extra raster offsets warn and preserve the built-in values and nodata mask exactly."""

        # Nonlinear surface to distinguish bilinear interpolation from triangulation
        rows, cols = np.indices((9, 9), dtype=float)
        values = rows * cols + np.sin(cols)
        values[4, 4] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 9, 1, 1), crs=32631, nodata=-9999)
        targets = (np.array([0.25, 2.3, 4.2, 7.6, 8.8]), np.array([8.75, 6.6, 4.7, 1.2, 0.1]))
        plain = ScipyInterpolator(method)
        configured = ScipyInterpolator(method, neighborhood=GridNeighbours(size=size))

        # Check oversized-window warning and agreement with default interpolation, including edges
        expected = raster.resample_at_points(targets, plain, nodata_handling=nodata_handling, as_array=True)
        with pytest.warns(UserWarning, match="extra GridNeighbours offsets are ignored") as recorded:
            result = raster.resample_at_points(targets, configured, nodata_handling=nodata_handling, as_array=True)
        assert len(recorded) == 1
        np.testing.assert_array_equal(result, expected)

    def test_resample_at_points__error_point_neighborhood_for_interpolator(self) -> None:
        """Checks that regular raster interpolation rejects a neighborhood expressed as a point search."""

        # Reject point-count neighborhood for regular bilinear interpolation
        raster = gu.Raster.from_array(np.ones((3, 3)), rio.transform.from_origin(0, 3, 1, 1), crs=32631)
        operator = Linear(neighborhood=PointNeighbours(k=4))
        with pytest.raises(ValueError, match="PointNeighbours applies to point sources"):
            raster.resample_at_points((1.2, 1.2), operator)

    @pytest.mark.parametrize("method,size", [("linear", 1), ("slinear", 1), ("pchip", 3)])
    def test_scipy_interpolator__error_incomplete_stencil(self, method: str, size: int) -> None:
        """Checks that regular interpolation rejects a window missing cells from its natural stencil."""

        with pytest.raises(ValueError, match="does not contain the natural"):
            ScipyInterpolator(method, neighborhood=GridNeighbours(size=size))  # type: ignore[arg-type]

    @pytest.mark.parametrize("offsets", [((1, 0),), ((0, 0), (0, 1), (1, 0))])
    def test_scipy_interpolator__error_missing_required_offset(self, offsets: tuple[tuple[int, int], ...]) -> None:
        """Checks that natural-stencil validation inspects offsets rather than only the window's bounding box."""

        method = "nearest" if len(offsets) == 1 else "linear"
        with pytest.raises(ValueError, match="does not contain the natural"):
            ScipyInterpolator(method, neighborhood=GridNeighbours(offsets))  # type: ignore[arg-type]

    @pytest.mark.parametrize("method", ["cubic", "quintic", "splinef2d"])
    def test_scipy_interpolator__error_fitted_spline_window(self, method: str) -> None:
        """Checks that fitted regular splines reject local windows even when the window is larger than their degree."""

        # Reject local window for splines fitted over complete raster
        with pytest.raises(ValueError, match="fits a spline over the grid"):
            ScipyInterpolator(method, neighborhood=GridNeighbours(size=5))  # type: ignore[arg-type]


class TestInterpolatorIrregular:
    """Test module for point interpolation, triangle coefficients, distance weights and coincident observations.

    Gridding backends are compared in test_execution.py; chunked gridding is covered in test_interface/test_gridding.py.
    """

    @pytest.mark.parametrize("operator,expected", [(Nearest(), 2.0), (Linear(), 5.0), (Cubic(), 5.0)])
    def test_evaluate__known_point_values(self, operator: Interpolator, expected: float) -> None:
        """Checks that nearest selects a point value while linear and cubic interpolation reproduce a plane."""

        # Three corners of plane 2 + 4*x + 8*y; nearest value two
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        target = np.array([0.25, 0.25])
        data = LocalData(
            values=np.array([2.0, 6.0, 10.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.arange(3),
            coordinates=coordinates,
            target=target,
            distances=np.linalg.norm(coordinates - target, axis=1),
        )
        result = operator.evaluate(data)

        # Allow cubic tolerance for iterative derivative estimates
        assert result == pytest.approx(expected, rel=0, abs=1e-6 if isinstance(operator, Cubic) else 1e-14)

    @pytest.mark.parametrize(
        "operator,expected", [(Nearest(), 2), (InverseDistance(power=1), 3), (InverseDistance(power=2), 2.4)]
    )
    def test_interpolator__distance_weights(self, operator: Interpolator, expected: float) -> None:
        """Checks that nearest and inverse-distance methods apply the expected weights to known distances."""

        # Distances 1 and 3: weights 3/4 and 1/4 (power 1), 9/10 and 1/10 (power 2)
        data = LocalData(
            values=np.array([2.0, 6.0]),
            valid=np.ones(2, dtype=bool),
            source_ids=np.arange(2),
            distances=np.array([1.0, 3.0]),
        )
        assert operator.evaluate(data) == pytest.approx(expected)

    def test_interpolator__linear_barycentric_coefficients(self) -> None:
        """Checks that Linear() returns the exact weights used for interpolation inside a triangle."""

        # Interior target with known barycentric weights in right triangle
        data = LocalData(
            values=np.array([2.0, 6.0, 10.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.arange(3),
            coordinates=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            target=np.array([0.25, 0.25]),
        )
        interpolator = Linear()
        coefficients = interpolator.coefficients(data)

        # Check barycentric weights and interpolated value
        assert coefficients is not None
        np.testing.assert_allclose(coefficients.weights, np.array([0.5, 0.25, 0.25]), rtol=0, atol=1e-15)
        assert interpolator.evaluate(data) == 5.0

    def test_linear__one_dimensional_coefficients(self) -> None:
        """Checks that one-dimensional interpolation uses the two bracketing observations in their original order."""

        # Midpoint target, distant observation in middle row to check weight order
        data = LocalData(
            values=np.array([8.0, 100.0, 2.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.array([7, 3, 5]),
            coordinates=np.array([[2.0], [9.0], [0.0]]),
            target=np.array([1.0]),
        )
        operator = Linear()
        coefficients = operator.coefficients(data)

        # Check coefficients in original observation order
        assert coefficients is not None
        np.testing.assert_array_equal(coefficients.weights, [0.5, 0.0, 0.5])
        assert operator.evaluate(data) == 5.0

    def test_inverse_distance__coincident_values(self) -> None:
        """Checks that coincident finite values share unit total weight and exclude more distant observations."""

        # Coincident observations with equal weight, distant observation with zero weight
        data = LocalData(
            values=np.array([2.0, 4.0, 100.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.arange(3),
            distances=np.array([0.0, 0.0, 1.0]),
        )
        coefficients = InverseDistance().coefficients(data)
        assert coefficients is not None
        np.testing.assert_array_equal(coefficients.weights, [0.5, 0.5, 0])
        assert InverseDistance().evaluate(data) == 3

    def test_interpolator__inverse_distance_rejects_support_weights(self) -> None:
        """Checks that point interpolation accepts sample weights but rejects area support weights."""

        # Unequal distances and sample weights
        data = LocalData(
            values=np.array([2.0, 4.0]),
            valid=np.ones(2, dtype=bool),
            source_ids=np.arange(2),
            distances=np.array([1.0, 2.0]),
            sample_weights=np.array([1.0, 3.0]),
        )
        interpolator = InverseDistance()

        # Check combined distance/sample weights: 1 and 3/4 before normalization
        assert interpolator.evaluate(data) == pytest.approx(20 / 7)

        # Reject area weights for point interpolation
        data_with_support = replace(data, support_weights=np.array([0.25, 0.75]))
        with pytest.raises(ValueError, match="does not accept support_weights"):
            interpolator.evaluate(data_with_support)


class TestPointInterpolationAccuracy:
    """Test module for IDW values, exact observations and triangulation on selected points.

    Gridding options and chunked execution are covered in test_interface/test_gridding.py.
    """

    @pytest.mark.parametrize("resampling", ["idw"])
    def test_grid_pc__circular_neighborhood(self, resampling: GriddingMethod) -> None:
        """Checks that IDW matches the analytical result for equidistant points."""

        # Two constant value columns place an equal pair of neighbors around the central column
        pc = gpd.GeoDataFrame(
            data={"z": [0.0, 10.0, 0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0, 0.0, 2.0], y=[0.0, 0.0, 1.0, 1.0]),
        )
        grid_coords = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))

        # A radius just over one pixel reaches both same-row neighbors at the center
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=1.1,
        )
        expected = np.array([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]])
        assert np.allclose(result, expected)

    def test_grid_pc__idw_distance_power_and_exact_points(self) -> None:
        """Checks that IDW follows its distance exponent and preserves exact source values."""

        # The first output cell coincides with a point while the middle cell has unequal distances
        pc = gpd.GeoDataFrame(
            data={"z": [0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 3.0], y=[0.0, 0.0]),
        )
        grid_coords = (np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0]))

        # Squared inverse distances give weights of one and one quarter at the inner columns
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling="idw",
            dist_nodata_pixel=2.1,
            distance_power=2,
        )
        assert np.allclose(result[1], [0.0, 2.0, 8.0, 10.0])

    @pytest.mark.parametrize("operator_type", [Linear, Cubic])
    @pytest.mark.parametrize(
        "neighborhood", [PointNeighbours(k=3), PointNeighbours(radius=2), PointNeighbours(k=3, radius=2)]
    )
    def test_grid__triangulation_uses_selected_points(
        self, operator_type: type[Interpolator], neighborhood: PointNeighbours
    ) -> None:
        """Checks that point triangulation uses the selected triangle and returns nodata outside its convex hull."""

        # The first three observations describe the plane Z=X+2Y; an excluded distant value violates that plane
        points = gu.PointCloud.from_xyz([0, 1, 0, 10], [0, 0, 1, 10], [0, 1, 2, 100], crs=32631)
        coords = (np.array([0.25, 0.75]), np.array([0.25, 0.75]))
        operator = operator_type(neighborhood=neighborhood)
        result = points.grid(grid_coords=coords, resampling=operator, nodata_handling="ignore")

        # Clough-Tocher's estimated derivatives reproduce this plane within SciPy's interpolation tolerance
        np.testing.assert_allclose(result.to_nanarray(), [[1.75, np.nan], [0.75, 1.25]], rtol=0, atol=1e-7)

    @pytest.mark.parametrize("operator_type", [Linear, Cubic])
    @pytest.mark.parametrize("count", [2, 3])
    def test_grid__insufficient_triangulation_returns_nodata(
        self, operator_type: type[Interpolator], count: int
    ) -> None:
        """Checks that too few or collinear selected points produce nodata without a Qhull failure."""

        # Three collinear points cannot define a two-dimensional triangle, and two points are also insufficient
        points = gu.PointCloud.from_xyz([0, 1, 2], [0, 0, 0], [2, 4, 8], crs=32631)
        operator = operator_type(neighborhood=PointNeighbours(k=count))
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.2, 0.2, 1, 1), crs=32631)
        result = points.grid(ref=reference, resampling=operator)
        assert np.isnan(result.to_nanarray()[0, 0])

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        "explicit_neighborhood,engine", [(False, "scipy"), (False, "numba"), (True, "scipy"), (True, "numba")]
    )
    def test_grid__idw_exact_value_excludes_noncontributing_nodata(
        self, explicit_neighborhood: bool, engine: Literal["scipy", "numba"]
    ) -> None:
        """Checks that IDW propagation ignores a missing neighbor when a finite point lies at the target."""

        # The output at X=0 coincides with a valid point; the point at X=1 is missing but lies inside the radius
        points = gu.PointCloud.from_xyz([0.0, 1.0], [0.0, 0.0], [2.0, np.nan], crs=32631)
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        neighborhood = PointNeighbours(k=2, radius=1.1) if explicit_neighborhood else None
        operator = InverseDistance(neighborhood=neighborhood)
        if engine == "numba":
            import_optional("numba")

        # An exact finite value has full weight; an exact missing value invalidates the other output cell
        result = points.grid(
            grid_coords=grid_coords,
            resampling=operator,
            dist_nodata_pixel=1.1,
            nodata_handling="propagate",
            engine=engine,
        )
        values = result.to_nanarray()
        assert values[1, 0] == 2.0
        assert np.isnan(values[1, 1])

    @pytest.mark.parametrize("explicit_neighborhood", [False, True])
    def test_grid__idw_gdal_masks_missing_nearest_with_either_neighborhood(self, explicit_neighborhood: bool) -> None:
        """Checks that IDW masks a missing nearest point with default or explicit neighbors."""

        # The missing point sits at X=0, while the finite point at X=0.5 can supply that output cell
        points = gu.PointCloud.from_xyz([0.0, 0.5], [0.0, 0.0], [np.nan, 6.0], crs=32631)
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        neighborhood = PointNeighbours(k=2, radius=1.1) if explicit_neighborhood else None
        operator = InverseDistance(neighborhood=neighborhood)

        # IDW calculates six from the finite point, then the nearest-point rule masks the result
        options: dict[str, Any] = {"grid_coords": grid_coords, "resampling": operator, "dist_nodata_pixel": 1.1}
        gdal = points.grid(**options, nodata_handling="gdal").to_nanarray()
        ignored = points.grid(**options, nodata_handling="ignore").to_nanarray()
        propagated = points.grid(**options, nodata_handling="propagate").to_nanarray()
        assert np.isnan(gdal[1, 0])
        assert ignored[1, 0] == 6.0
        assert np.isnan(propagated[1, 0])


class TestRasterInterpolationAccuracy:
    """Test module for known raster values and agreement between SciPy interpolation methods.

    Point formats, coordinates and chunked sampling are covered in test_interface/test_resampling.py.
    """

    landsat_b4_path = gu.examples.get_path_test("everest_landsat_b4")
    aster_dem_path = gu.examples.get_path_test("exploradores_aster_dem")

    def test_reproject__nearest_operator_uses_raster_cells(self) -> None:
        """Checks that Nearest() uses raster cells without needing a point or grid neighbor choice."""

        # Reproject a finite raster onto its own grid through the Nearest operator
        values = np.arange(9, dtype=float).reshape(3, 3)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 3, 1, 1), crs=4326)
        result = raster.reproject(raster, resampling=Nearest())

        # Each target lies on its matching source cell, so all values should be unchanged
        assert result is not None
        np.testing.assert_array_equal(result.to_nanarray(), values)

    @pytest.mark.parametrize("method,order", [("nearest", 0), ("linear", 1)])
    def test_interp_at_points__unequal_xy_spacing(self, method: str, order: int) -> None:
        """
        Checks that interpolation gives equal result on both map_coordinates/interpn SciPy methods when X and Y pixel
        sizes differ (ensures the fast map_coordinates() is accurate with different X/Y spacing).
        """

        # Create raster with unequal pixel sizes
        values = np.array([[2, 11, 5, 19], [13, 7, 17, 3], [23, 29, 31, 37], [41, 43, 47, 53]], dtype=float)
        transform = rio.transform.from_origin(100, 200, 2, 5)
        raster = gu.Raster.from_array(values, transform, crs=32631, area_or_point="Area")
        rows = np.array([0.25, 1.3, 2.65])
        columns = np.array([0.4, 2.1, 1.2])
        x = transform.c + columns * transform.a
        y = transform.f + rows * transform.e

        # Compare map_coords/interpn/interp_at_points
        y_axis = transform.f + np.arange(values.shape[0]) * transform.e
        x_axis = transform.c + np.arange(values.shape[1]) * transform.a
        from_indices = map_coordinates(values, (rows, columns), order=order, mode="nearest", prefilter=False)
        from_coordinates = interpn((y_axis, x_axis), values, np.column_stack((y, x)), method=method)
        sampled = raster.interp_at_points((x, y), method=method, as_array=True)

        # All should agree closely
        np.testing.assert_allclose(from_indices, from_coordinates, rtol=0, atol=1e-12)
        np.testing.assert_allclose(sampled, from_coordinates, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("tag_aop", [None, "Area", "Point"])
    @pytest.mark.parametrize("shift_aop", [True, False])
    def test_interp_points__synthetic(self, tag_aop: Literal["Area", "Point"] | None, shift_aop: bool) -> None:
        """
        Checks that interp_points() matches known values and SciPy methods on synthetic data.

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

        raster_points = raster.interp_at_points(
            (points_x, points_y), method="nearest", shift_area_or_point=shift_aop, as_array=True
        )
        raster_points_lin = raster.interp_at_points(
            (points_x, points_y), method="linear", shift_area_or_point=shift_aop, as_array=True
        )
        raster_points_interpn = raster.interp_at_points(
            (points_x, points_y),
            method="nearest",
            force_scipy_function="interpn",
            shift_area_or_point=shift_aop,
            as_array=True,
        )
        raster_points_interpn_lin = raster.interp_at_points(
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
        raster_points_in = raster.interp_at_points(
            (points_x_in, points_y_in), method="linear", shift_area_or_point=shift_aop, as_array=True
        )
        raster_points_in_interpn = raster.interp_at_points(
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

        # Select points beyond the outer half pixels for every pixel interpretation
        points_out_xy = raster.ij2xy([-2, 1, 4, 1], [1, -2, 1, 4], shift_area_or_point=shift_aop)
        with pytest.warns(UserWarning, match="All provided points were outside of raster bounds"):
            raster_points_out = raster.interp_at_points(points_out_xy, shift_area_or_point=shift_aop, as_array=True)
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
            raster_points_mapcoords = raster.interp_at_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="map_coordinates",
                shift_area_or_point=shift_aop,
                as_array=True,
            )
            raster_points_interpn = raster.interp_at_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="interpn",
                shift_area_or_point=shift_aop,
                as_array=True,
            )

            # Not exactly equal in floating point precision since changes in Scipy 1.13.0,
            # see https://github.com/GlacioHack/geoutils/issues/533
            assert np.allclose(raster_points_mapcoords, raster_points_interpn)

        # Nearest and linear include the outer half pixels; splines keep their stricter coordinate bounds
        index_x_edge_rand = [-0.5, -0.5, -0.5, 25, 25, 49.5, 49.5, 49.5]
        index_y_edge_rand = [-0.5, 25, 49.5, -0.5, 49.5, -0.5, 25, 49.5]

        points_x_rand, points_y_rand = raster.ij2xy(
            i=index_x_edge_rand, j=index_y_edge_rand, shift_area_or_point=shift_aop
        )

        # Test across all methods
        for method in ["nearest", "linear", "cubic", "quintic"]:
            raster_points_mapcoords_edge = raster.interp_at_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="map_coordinates",
                shift_area_or_point=shift_aop,
                as_array=True,
            )
            raster_points_interpn_edge = raster.interp_at_points(
                (points_x_rand, points_y_rand),
                method=method,
                force_scipy_function="interpn",
                shift_area_or_point=shift_aop,
                as_array=True,
            )

            finite = (
                [True, True, False, True, False, False, False, False]
                if method in {"nearest", "linear"}
                else [False] * 8
            )
            np.testing.assert_array_equal(np.isfinite(raster_points_mapcoords_edge), finite)
            np.testing.assert_array_equal(np.isfinite(raster_points_interpn_edge), finite)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    @pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"])  # type: ignore
    def test_interp_points__real(
        self, example: str, method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"]
    ) -> None:
        """
        Checks that interpolation methods agree on real data, including nodata.

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
        val = r.interp_at_points((x, y), method=method, force_scipy_function="map_coordinates", as_array=True)[0]
        val_img = r.to_nanarray()[itest, jtest]
        # For a point exactly at a grid coordinate, only nearest and linear will match
        # (cubic modifies values at a grid coordinate)
        if method in ["nearest", "linear"]:
            assert val_img == pytest.approx(val, nan_ok=True)

        # Check the result is exactly the same for both methods
        val2 = r.interp_at_points((x, y), method=method, force_scipy_function="interpn", as_array=True)[0]
        assert val2 == pytest.approx(val, nan_ok=True)

        # Check that interp convert to latlon
        lat, lon = gu.projtools.reproject_to_latlon((x, y), in_crs=r.crs)
        val_latlon = r.interp_at_points((lat, lon), method=method, input_latlon=True, as_array=True)[0]
        assert val == pytest.approx(val_latlon, abs=0.0001, nan_ok=True)

        # 2/ Test for multiple points
        i = np.random.default_rng(42).integers(1, 49, size=10)
        j = np.random.default_rng(42).integers(1, 49, size=10)
        x, y = r.ij2xy(i, j)
        vals = r.interp_at_points((x, y), method=method, force_scipy_function="map_coordinates", as_array=True)
        vals2 = r.interp_at_points((x, y), method=method, force_scipy_function="interpn", as_array=True)

        assert np.array_equal(vals, vals2, equal_nan=True)

        # 3/ Test return_interpolator is consistent with above
        interp = _resample_at_points(
            r,
            points=(x, y),
            method=method,
            return_interpolator=True,
        )
        vals3 = interp((y, x))
        vals3 = np.array(np.atleast_1d(vals3), dtype=np.float32)

        assert np.array_equal(vals2, vals3, equal_nan=True)


class TestPointInterpolationEngines:
    """Test module for SciPy/Numba agreement on point interpolation and radius boundaries.

    Reducer engine comparisons are in test_reducer.py; chunked gridding is in test_interface/test_gridding.py.
    """

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        "resampling",
        ["nearest", "idw"],
    )
    def test_grid_pc__engine(self, resampling: GriddingMethod) -> None:
        """Checks that the SciPy and Numba engines give the same gridded values."""

        import_optional("numba")

        # Uneven point values and positions exercise distance choices and every accumulation
        pc = gpd.GeoDataFrame(
            data={"z": [1.0, 4.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.2, 3.0], y=[0.0, 1.0, 0.0]),
        )
        grid_coords = (np.arange(4, dtype=float), np.arange(2, dtype=float))
        scipy_result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=2,
            engine="scipy",
        )

        # The explicit Numba engine follows the same interface as elsewhere in GeoUtils and xDEM
        numba_result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=2,
            engine="numba",
        )
        assert np.allclose(scipy_result, numba_result, equal_nan=True)

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        ("resampling", "expected"),
        [("nearest", np.array([[4.0, 4.0, np.nan]])), ("idw", np.array([[4.0, 4.0, np.nan]]))],
    )
    def test_grid_pc__engines_include_support_boundary(self, resampling: GriddingMethod, expected: NDArrayNum) -> None:
        """Checks that both Numba/SciPy engines include cells on the support radius and exclude cells beyond it."""

        import_optional("numba")
        pc = gpd.GeoDataFrame(data={"z": [4.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        for engine in ("scipy", "numba"):
            result, _ = _grid_pointcloud(
                pc,
                grid_coords=(np.arange(3, dtype=float), np.array([0.0])),
                grid_res=(1.0, 1.0),
                data_column_name="z",
                resampling=resampling,
                dist_nodata_pixel=1,
                engine=engine,
            )
            assert np.allclose(result, expected, equal_nan=True)

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    def test_grid_pc__engines_average_duplicate_exact_idw_points(self) -> None:
        """Checks that both Numba/SciPy engines average duplicate exact IDW points even when min_points is unmet."""

        import_optional("numba")
        # Process a weighted neighbor before two exact points to ensure exact values replace it
        pc = gpd.GeoDataFrame(
            data={"z": [100.0, 2.0, 8.0]},
            geometry=gpd.points_from_xy(x=[1.0, 0.0, 0.0], y=[0.0, 0.0, 0.0]),
        )
        for engine in ("scipy", "numba"):
            result, _ = _grid_pointcloud(
                pc,
                grid_coords=(np.array([0.0]), np.array([0.0])),
                grid_res=(1.0, 1.0),
                data_column_name="z",
                resampling="idw",
                dist_nodata_pixel=1.1,
                min_points=4,
                engine=engine,
            )
            assert result[0, 0] == pytest.approx(5.0)

    @pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba")
    @pytest.mark.parametrize(
        "operator_type",
        [Nearest, InverseDistance],
    )
    @pytest.mark.parametrize(
        "neighborhood", [PointNeighbours(k=3), PointNeighbours(radius=1.1), PointNeighbours(k=3, radius=1.1)]
    )
    @pytest.mark.parametrize("nodata_handling", ["ignore", "gdal", "propagate"])
    def test_grid__numba_matches_scipy_for_explicit_neighborhoods(
        self,
        operator_type: type[Interpolator],
        neighborhood: PointNeighbours,
        nodata_handling: NodataChoice,
    ) -> None:
        """Checks that every supported Numba method uses the same selected points and nodata rule as SciPy."""

        import_optional("numba")

        # Include coincident observations, a missing value, radius boundaries and an empty output neighborhood
        points = gu.PointCloud.from_xyz([0, 0, 1, 2, 5], [0, 0, 0, 1, 2], [2, 4, np.nan, 8, 20], crs=32631)
        grid_coords = (np.arange(7, dtype=float), np.arange(4, dtype=float))
        operator = operator_type(neighborhood=neighborhood)
        options: dict[str, Any] = {
            "grid_coords": grid_coords,
            "resampling": operator,
            "nodata_handling": nodata_handling,
        }

        # Both engines must select the same observations, including at exact point locations
        expected = points.grid(**options, engine="scipy")
        result = points.grid(**options, engine="numba")
        np.testing.assert_allclose(result.to_nanarray(), expected.to_nanarray(), rtol=1e-14, atol=1e-14)


class TestRasterInterpolationGDAL:
    """Test module for regular interpolation values and nodata masks compared with GDAL.

    Reprojection grids and chunked execution are covered in test_raster/test_transformations_raster.py.
    """

    @pytest.mark.parametrize(
        ("method", "resampling"),
        [("nearest", rio.enums.Resampling.nearest), ("linear", rio.enums.Resampling.bilinear)],
    )
    @pytest.mark.parametrize("shift", [(1.0, 1.0), (0.2, 0.2), (0.5, 0.5), (-0.2, 0.35), (-0.5, -0.5)])
    def test_interpolate_array__gdal_nodata(
        self,
        method: Literal["nearest", "linear"],
        resampling: rio.enums.Resampling,
        shift: tuple[float, float],
    ) -> None:
        """Checks that interpolation matches GDAL values and invalid cells for integral and subpixel shifts."""

        # Separate invalid regions exercise isolated cells, edges and neighboring invalid cells
        source = np.arange(81, dtype=np.float32).reshape(9, 9)
        source[4, 4] = np.nan
        source[1:3, 6] = np.nan
        src_transform = rio.transform.from_origin(0, 9, 1, 1)
        dst_transform = Affine.translation(*shift) * src_transform

        # Rasterio calls the GDAL resampler with an explicit sentinel for source nodata
        expected = np.full(source.shape, np.nan, dtype=np.float32)
        with silence_rasterio_message(param_name="SCALE"):
            rio.warp.reproject(
                np.where(np.isfinite(source), source, -9999),
                expected,
                src_transform=src_transform,
                dst_transform=dst_transform,
                src_crs=4326,
                dst_crs=4326,
                src_nodata=-9999,
                dst_nodata=np.nan,
                resampling=resampling,
                XSCALE=1,
                YSCALE=1,
            )

        # The default interpolation policy must reproduce both GDAL values and its nodata mask
        actual = _interpolate_array(
            source,
            src_transform=src_transform,
            dst_transform=dst_transform,
            method=method,
        )
        assert np.array_equal(np.isnan(actual), np.isnan(expected))
        assert np.allclose(actual, expected, equal_nan=True, atol=2e-5)

    @pytest.mark.parametrize("resolution", [0.5, 1.5, 2.0])
    def test_interpolate_array__gdal_resolution(self, resolution: float) -> None:
        """Checks that bilinear interpolation matches GDAL nodata behavior while changing output resolution."""

        # Use several nodata shapes because their contributions change between resolutions
        source = np.arange(81, dtype=np.float32).reshape(9, 9)
        source[4, 4] = np.nan
        source[1:3, 6] = np.nan
        src_transform = rio.transform.from_origin(0, 9, 1, 1)
        dst_shape = (int(np.ceil(source.shape[0] / resolution)), int(np.ceil(source.shape[1] / resolution)))
        dst_transform = rio.transform.from_origin(0, 9, resolution, resolution)

        # Compare the complete array so both internal nodata and outer footprint behavior are covered
        expected = np.full(dst_shape, np.nan, dtype=np.float32)
        with silence_rasterio_message(param_name="SCALE"):
            rio.warp.reproject(
                np.where(np.isfinite(source), source, -9999),
                expected,
                src_transform=src_transform,
                dst_transform=dst_transform,
                src_crs=4326,
                dst_crs=4326,
                src_nodata=-9999,
                dst_nodata=np.nan,
                resampling=rio.enums.Resampling.bilinear,
                XSCALE=1,
                YSCALE=1,
            )
        actual = _interpolate_array(
            source,
            src_transform=src_transform,
            dst_transform=dst_transform,
            dst_shape=dst_shape,
        )
        assert np.array_equal(np.isnan(actual), np.isnan(expected))
        assert np.allclose(actual, expected, equal_nan=True, atol=2e-5)

    @pytest.mark.parametrize(
        ("method", "resampling"),
        [
            ("nearest", rio.enums.Resampling.nearest),
            ("linear", rio.enums.Resampling.bilinear),
            ("cubic", rio.enums.Resampling.cubic),
        ],
    )
    @pytest.mark.parametrize("shift", [(0.2, 0.2), (0.5, 0.5), (-0.5, -0.5)])
    def test_interp_points__gdal_nodata_matches_reproject(
        self,
        method: Literal["nearest", "linear", "cubic"],
        resampling: rio.enums.Resampling,
        shift: tuple[float, float],
    ) -> None:
        """Checks that default point interpolation matches the nodata mask from GDAL reprojection in the same CRS."""

        # Create isolated and adjacent nodata cells so several kernel neighborhoods meet invalid values
        source = np.arange(121, dtype=np.float32).reshape(11, 11)
        source[5, 5] = np.nan
        source[2:4, 7] = np.nan
        source_transform = rio.transform.from_origin(0, 11, 1, 1)
        source_raster = gu.Raster.from_array(source, transform=source_transform, crs=4326, nodata=-9999)

        # Shift an interior seven by seven grid so every destination remains inside the source footprint
        reference_transform = rio.transform.from_origin(1 + shift[0], 10 + shift[1], 1, 1)
        reference = gu.Raster.from_array(
            np.zeros((7, 7), dtype=np.float32),
            transform=reference_transform,
            crs=4326,
            nodata=-9999,
        )
        expected = source_raster.reproject(reference, resampling=resampling).to_nanarray()
        rows, columns = np.indices(reference.shape)
        x, y = reference.ij2xy(rows.ravel(), columns.ravel())

        # Sample the same destination centers and compare the complete validity mask
        actual = source_raster.interp_at_points((x, y), method=method, as_array=True).reshape(reference.shape)
        assert np.array_equal(np.isnan(actual), np.isnan(expected))
        if method in ("nearest", "linear"):
            assert np.allclose(actual, expected, equal_nan=True, atol=2e-5)

    def test_reproject__linear_operator_matches_bilinear(self) -> None:
        """Checks that Linear() uses the same source cells and weights as bilinear reprojection."""

        # Shift the output grid by a quarter cell so each result combines four different source values
        values = np.arange(16, dtype=float).reshape(4, 4)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 4, 1, 1), crs=4326)
        reference = gu.Raster.from_array(
            np.zeros((2, 2)), transform=rio.transform.from_origin(0.25, 3.75, 1, 1), crs=4326
        )

        # Compare the operator result with Rasterio's bilinear result at those same output centers
        expected = raster.reproject(reference, resampling="bilinear")
        actual = raster.reproject(reference, resampling=Linear())
        assert expected is not None and actual is not None
        np.testing.assert_allclose(actual.to_nanarray(), expected.to_nanarray())


class TestPointInterpolationGDAL:
    """Test module for point interpolation compared with GDAL on circular and anisotropic neighborhoods."""

    @pytest.mark.parametrize(("geoutils_method", "gdal_algorithm"), [("nearest", "nearest"), ("linear", "linear")])
    def test_grid_pc__gdal_interpolation(
        self, geoutils_method: GriddingMethod, gdal_algorithm: str, tmp_path: Path
    ) -> None:
        """Checks that nearest and linear interpolation match complete GDAL outputs for an irregular point cloud."""

        # Unequal positions and values expose axis rescaling and nearest-neighbor differences
        points = gpd.GeoDataFrame(
            {"z": [2.0, 8.0, 4.0, 10.0, 6.0]},
            geometry=gpd.points_from_xy(x=[0.0, 37.0, 4.0, 40.0, 17.0], y=[0.0, 0.2, 3.6, 4.0, 2.3]),
            crs=32631,
        )
        point_file = tmp_path / "interpolation-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        grid_coords = (np.arange(0, 50, 10, dtype=float), np.arange(5, dtype=float))

        # Infinite nearest support and linear interpolation without extrapolation match GDAL radius zero
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=geoutils_method,
            dist_nodata_pixel=float("inf"),
            engine="scipy",
        )
        output_file = tmp_path / f"gdal-{gdal_algorithm}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm=gdal_algorithm,  # type: ignore[arg-type]
            bounds=(-5.0, -0.5, 45.0, 4.5),
            shape=(5, 5),
            radius=(0.0, 0.0),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    @pytest.mark.parametrize(
        ("geoutils_method", "gdal_algorithm"),
        [("idw", "invdist")],
    )
    def test_grid_pc__gdal_circular_methods(
        self, geoutils_method: GriddingMethod, gdal_algorithm: str, tmp_path: Path
    ) -> None:
        """Checks that circular methods match GDAL values and nodata cells."""

        # Two unequal values exercise all statistics while an invalid value checks GDAL's omission rule
        points = gpd.GeoDataFrame(
            {"z": [2.0, np.nan, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / "circular-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        x_coords = np.arange(5, dtype=float)
        y_coords = np.array([0.0])

        # GeoUtils expresses the support ellipse in output pixels
        # Use finite points only to match GDAL's point gridding (including IDW)
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(x_coords, y_coords),
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling=geoutils_method,
            dist_nodata_pixel=1.1,
            nodata_handling="ignore",
        )

        # GDAL receives the equivalent support in coordinate units and the same cell centers
        output_file = tmp_path / f"gdal-{gdal_algorithm}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm=gdal_algorithm,  # type: ignore[arg-type]
            bounds=(-0.5, -0.5, 4.5, 0.5),
            shape=(1, 5),
            radius=(1.1, 1.1),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    def test_grid_pc__gdal_idw_minimum_points(self, tmp_path: Path) -> None:
        """Checks that exact IDW points take precedence over minimum neighbor counts like GDAL."""

        # Exact edge cells have one neighbor while the central cell has two
        points = gpd.GeoDataFrame(
            {"z": [2.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0], y=[0.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / "idw-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(np.arange(3, dtype=float), np.array([0.0])),
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling="idw",
            dist_nodata_pixel=1.1,
            min_points=2,
        )

        # Use the same local support and minimum count in the GDAL reference
        output_file = tmp_path / "gdal-idw-min-points.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm="invdist",
            bounds=(-0.5, -0.5, 2.5, 0.5),
            shape=(1, 3),
            radius=(1.1, 1.1),
            min_points=2,
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    @pytest.mark.parametrize(
        "engine",
        ["scipy", pytest.param("numba", marks=pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba"))],
    )
    def test_grid_pc__gdal_idw_anisotropic_grid(self, engine: str, tmp_path: Path) -> None:
        """Checks that IDW weights use coordinate distances on an anisotropic output grid like GDAL."""

        if engine == "numba":
            import_optional("numba")

        # One X pixel is ten times larger than one Y pixel so scaled and coordinate distances differ
        points = gpd.GeoDataFrame(
            {"z": [0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 10.0], y=[1.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / f"anisotropic-idw-{engine}.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(np.arange(0, 30, 10, dtype=float), np.arange(3, dtype=float)),
            data_column_name="z",
            resampling="idw",
            dist_nodata_pixel=1.1,
            engine=engine,  # type: ignore[arg-type]
        )

        # GDAL receives the same support radius converted from pixels to coordinate units
        output_file = tmp_path / f"gdal-anisotropic-idw-{engine}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm="invdist",
            bounds=(-5.0, -0.5, 25.0, 2.5),
            shape=(3, 3),
            radius=(11.0, 1.1),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    @pytest.mark.parametrize(
        ("resampling", "raster_resampling"),
        [
            ("nearest", rio.enums.Resampling.nearest),
            ("linear", rio.enums.Resampling.bilinear),
            ("cubic", rio.enums.Resampling.cubic),
        ],
    )
    @pytest.mark.parametrize("shift", [(0.2, 0.2), (0.5, 0.5), (-0.5, -0.5)])
    def test_grid_pc__gdal_nodata_matches_reproject(
        self,
        resampling: GriddingMethod,
        raster_resampling: rio.enums.Resampling,
        shift: tuple[float, float],
    ) -> None:
        """Checks that default point gridding matches the nodata mask from GDAL reprojection in the same CRS."""

        # Represent every source raster cell as a point, including three observations with invalid values
        source = np.arange(121, dtype=np.float32).reshape(11, 11)
        source[5, 5] = np.nan
        source[2:4, 7] = np.nan
        source_transform = rio.transform.from_origin(0, 11, 1, 1)
        source_raster = Raster.from_array(source, transform=source_transform, crs=4326, nodata=-9999)
        rows, columns = np.indices(source.shape)
        source_x, source_y = source_raster.ij2xy(rows.ravel(), columns.ravel())
        point_cloud = gpd.GeoDataFrame(
            data={"z": source.ravel()},
            geometry=gpd.points_from_xy(source_x, source_y),
            crs=source_raster.crs,
        )
        # Shuffle source rows to check that ties at half-pixel distances do not depend on input order
        point_cloud = point_cloud.sample(frac=1, random_state=42)

        # Shift an interior grid to compare the nearest source selected, away from the edges of the point cloud
        reference = Raster.from_array(
            np.zeros((7, 7), dtype=np.float32),
            transform=rio.transform.from_origin(1 + shift[0], 10 + shift[1], 1, 1),
            crs=4326,
            nodata=-9999,
        )
        expected = source_raster.reproject(reference, resampling=raster_resampling).to_nanarray()

        # Grid the points at the same destination centers and compare the complete validity mask
        actual, _ = _grid_pointcloud(
            point_cloud,
            grid_coords=reference.coords(grid=False),
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=np.inf,
        )
        assert np.array_equal(np.isnan(actual), np.isnan(expected))

        # Ignoring missing values fills the cells GDAL masks; propagation must mask at least those same cells
        ignored, _ = _grid_pointcloud(
            point_cloud,
            grid_coords=reference.coords(grid=False),
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=np.inf,
            nodata_handling="ignore",
        )
        propagated, _ = _grid_pointcloud(
            point_cloud,
            grid_coords=reference.coords(grid=False),
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=np.inf,
            nodata_handling="propagate",
        )
        if shift == (0.2, 0.2):
            assert np.all(np.isfinite(ignored[np.isnan(expected)]))
            assert np.all(np.isnan(propagated)[np.isnan(expected)])


class TestPointInterpolationUncertainty:
    """Test module for point interpolation values and uncertainty across numerical engines."""

    @pytest.mark.parametrize(
        "operator_type",
        [Nearest, InverseDistance],
    )
    @pytest.mark.parametrize(
        "engine",
        ["scipy", pytest.param("numba", marks=pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba"))],
    )
    def test_grid__zero_error_with_explicit_neighborhood(
        self, operator_type: type[Interpolator], engine: Literal["scipy", "numba"]
    ) -> None:
        """Checks that zero error reproduces the gridded values for every engine and supported neighborhood method."""

        if engine == "numba":
            import_optional("numba")

        # All targets have three finite neighbors with distinct distances
        points = gu.PointCloud.from_xyz([0, 0.8, 2.1, 3.3], [0, 0.1, 2.2, 0.3], [2, 4, 8, 16], crs=32631)
        operator = operator_type(neighborhood=PointNeighbours(k=3))
        grid_coords = (np.array([0.2, 1.2]), np.array([0.3, 1.3]))
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Small rounding differences can arise from the compiled weighted sums
        result = points.grid(
            grid_coords=grid_coords,
            resampling=operator,
            engine=engine,
            nodata_handling="ignore",
            error_structure=errors,
        )
        summary = gu.uncertainty.propagate(
            points.grid,
            error_structure=errors,
            operation_kwargs={
                "grid_coords": grid_coords,
                "resampling": operator,
                "engine": engine,
                "nodata_handling": "ignore",
            },
            method="numerical",
            n_samples=2,
            random_state=4,
        )
        np.testing.assert_allclose(
            summary.mean.to_nanarray().reshape(-1), result.to_nanarray().reshape(-1), rtol=1e-14, atol=1e-14
        )
        np.testing.assert_array_equal(summary.variance.to_nanarray().reshape(-1), np.zeros(4))

    @pytest.mark.parametrize(
        "engine",
        ["scipy", pytest.param("numba", marks=pytest.mark.skipif(find_spec("numba") is None, reason="Requires numba"))],
    )
    def test_grid__idw_exact_match_overrides_min_points(self, engine: Literal["scipy", "numba"]) -> None:
        """Checks that an exact IDW observation overrides min_points with or without an explicit neighborhood."""

        if engine == "numba":
            import_optional("numba")

        # Only the target at X=0 coincides with the sole observation
        points = gu.PointCloud.from_xyz([0], [0], [7], crs=32631)
        coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        for neighborhood in (None, PointNeighbours(k=1, radius=2)):
            result = points.grid(
                grid_coords=coords,
                resampling=InverseDistance(neighborhood=neighborhood),
                min_points=3,
                dist_nodata_pixel=2,
                engine=engine,
                nodata_handling="ignore",
                error_structure=errors,
            )
            summary = gu.uncertainty.propagate(
                points.grid,
                error_structure=errors,
                operation_kwargs={
                    "grid_coords": coords,
                    "resampling": InverseDistance(neighborhood=neighborhood),
                    "min_points": 3,
                    "dist_nodata_pixel": 2,
                    "engine": engine,
                    "nodata_handling": "ignore",
                },
            )

            # The exact target uses unit weight and preserves variance four; other targets lack enough points
            np.testing.assert_array_equal(result.to_nanarray(), [[np.nan, np.nan], [7, np.nan]])
            np.testing.assert_array_equal(summary.variance.to_nanarray().reshape(-1), [np.nan, np.nan, 4, np.nan])


class TestRegularInterpolationUncertainty:
    """Test module for uncertainty propagation through regular interpolation methods."""

    def test_reproject__propagates_bilinear_uncertainty(self) -> None:
        """Checks that requesting uncertainty returns the same bilinear values as GDAL and uses the same weights."""

        # Shift one destination center one-quarter pixel right and down from source row/column one
        values = np.arange(16, dtype=np.float64).reshape(4, 4)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 4, 1, 1), crs=4326, nodata=-9999)
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            rio.transform.from_origin(0.25, 3.75, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # The first tuple item remains the ordinary Rasterio result
        expected = raster.reproject(reference, resampling="bilinear")
        nominal = raster.reproject(resampling="bilinear", ref=reference, error_structure=source_error)
        summary = gu.uncertainty.propagate(
            raster.reproject,
            error_structure=source_error,
            operation_kwargs={"resampling": "bilinear", "ref": reference},
        )
        assert expected is not None
        assert nominal.raster_equal(expected, strict_masked=False)
        np.testing.assert_array_equal(summary.estimate.to_nanarray().reshape(-1), expected.to_nanarray().reshape(-1))

        # The output variance is four times the sum of squared weights (9/16, 3/16, 3/16 and 1/16)
        np.testing.assert_allclose(summary.variance.to_nanarray().reshape(-1), [1.5625])

    @pytest.mark.parametrize("method", ["nearest", "linear", "slinear", "cubic", "quintic", "pchip", "splinef2d"])
    @pytest.mark.parametrize("missing", [False, True])
    @pytest.mark.parametrize("nodata_handling", ["ignore", "gdal", "propagate"])
    def test_resample_at_points__zero_error_uses_original_method(
        self, method: ScipyInterpolationMethod, missing: bool, nodata_handling: NodataChoice
    ) -> None:
        """Checks that zero source error reproduces each regular interpolation method during numerical propagation."""

        # Random finite values make a change from regular splines to local triangulation visible
        values = np.random.default_rng(4).normal(size=(8, 9))
        if missing:
            values[4, 4] = np.nan
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), crs=32631, nodata=-9999)
        # The final target stays outside the widest nodata mask so every method has a finite result
        targets = (np.array([2.3, 3.7, 5.2, 0.2]), np.array([3.4, 4.6, 5.1, 7.8]))
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Numerical propagation must use exactly the method used for the nominal values
        result = raster.resample_at_points(
            as_array=True,
            nodata_handling=nodata_handling,
            points=targets,
            method=ScipyInterpolator(method),
            error_structure=errors,
        )
        summary = gu.uncertainty.propagate(
            raster.resample_at_points,
            error_structure=errors,
            operation_kwargs={
                "as_array": True,
                "nodata_handling": nodata_handling,
                "points": targets,
                "method": ScipyInterpolator(method),
            },
            method="numerical",
            n_samples=2,
            random_state=4,
        )
        np.testing.assert_allclose(summary.mean, result, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(summary.variance, np.where(np.isfinite(result), 0, np.nan))

    @pytest.mark.parametrize("method", ["cubic", "quintic", "splinef2d"])
    def test_resample_at_points__distant_error_changes_spline(self, method: ScipyInterpolationMethod) -> None:
        """Checks that numerical propagation refits the spline when an input outside a local stencil changes."""

        # A deterministic error at the first corner affects fitted coefficients beyond its neighboring cells
        values = np.random.default_rng(9).normal(size=(8, 8))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), crs=32631)
        targets = (np.array([3.3]), np.array([4.2]))
        mean = pd.Series(np.zeros(values.size))
        mean.iloc[0] = 10
        errors = gu.ErrorStructure.from_gaussian(pd.DataFrame(np.zeros((64, 64))), mean=mean)

        # Compare propagation with a complete refit of an independently changed raster
        nominal = raster.resample_at_points(
            as_array=True, points=targets, method=ScipyInterpolator(method), error_structure=errors
        )
        summary = gu.uncertainty.propagate(
            raster.resample_at_points,
            error_structure=errors,
            operation_kwargs={"as_array": True, "points": targets, "method": ScipyInterpolator(method)},
            method="numerical",
            n_samples=2,
            random_state=5,
        )
        changed_values = values.copy()
        changed_values[0, 0] += 10
        changed = gu.Raster.from_array(changed_values, raster.transform, crs=raster.crs)
        expected = changed.interp_at_points(targets, method=method, as_array=True)
        assert abs(expected[0] - nominal[0]) > 1e-7
        np.testing.assert_allclose(summary.mean, expected, rtol=0, atol=1e-12)


class TestRasterInterpolationNodata:
    """Test module for interpolation near missing values and agreement across nodata distances.

    Spatial mask helpers are covered in test_nodata.py; chunked sampling is in test_interface/test_resampling.py.
    """

    landsat_b4_path = gu.examples.get_path_test("everest_landsat_b4")
    aster_dem_path = gu.examples.get_path_test("exploradores_aster_dem")

    def test_interp_points__nodata_policies(self) -> None:
        """Checks that GDAL-compatible, ignored and propagated nodata rules produce their expected masks."""

        # Two fractional samples receive weight from the invalid central cell
        source = np.arange(25, dtype=np.float32).reshape(5, 5)
        source[2, 2] = np.nan
        raster = gu.Raster.from_array(
            source,
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=-9999,
        )
        x, y = raster.ij2xy(i=np.array([2.25, 1.4]), j=np.array([2.25, 1.4]))

        # Ignoring nodata retains finite contributions while propagation rejects every affected sample
        ignored = raster.interp_at_points((x, y), as_array=True, nodata_handling="ignore")
        gdal = raster.interp_at_points((x, y), as_array=True)
        propagated = raster.interp_at_points((x, y), as_array=True, nodata_handling="propagate")
        assert np.all(np.isfinite(ignored))
        assert np.array_equal(np.isnan(gdal), np.array([True, False]))
        assert np.all(np.isnan(propagated))

        with pytest.raises(ValueError, match="nodata_handling must be"):
            raster.interp_at_points((x, y), as_array=True, nodata_handling="invalid")  # type: ignore[arg-type]

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    @pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"])
    @pytest.mark.parametrize("dist", ["half_order_up", "half_order_down", 0, 1, 5])
    def test_interp_point__nodata_propag(
        self,
        example: str,
        method: Literal["nearest", "linear", "cubic", "quintic", "slinear", "pchip", "splinef2d"],
        dist: Literal["half_order_up", "half_order_down"] | int,
    ) -> None:
        """
        Checks that interp_points() propagates nodata to the requested distance.

        We create artificial NaNs at certain pixels of real rasters (one with integer type, one floating type), then
        verify that interpolated values propagate these NaNs at the right distances for all methods, and across all
        nodata propagation distances.

        We also assess the precision of interpolation under the "half_order_up" nodata propagation rule by
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
        vals = r.interp_at_points(
            (x, y), method=method, force_scipy_function="map_coordinates", nodata_handling=dist, as_array=True
        )
        vals2 = r.interp_at_points(
            (x, y), method=method, force_scipy_function="interpn", nodata_handling=dist, as_array=True
        )

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # Same check for random coordinates within half a pixel of the above coordinates (falling exactly on grid
        # points)
        xoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))
        yoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))

        vals = r.interp_at_points(
            (x + xoffset, y + yoffset),
            method=method,
            force_scipy_function="map_coordinates",
            nodata_handling=dist,
            as_array=True,
        )
        vals2 = r.interp_at_points(
            (x + xoffset, y + yoffset),
            method=method,
            force_scipy_function="interpn",
            nodata_handling=dist,
            as_array=True,
        )

        assert all(np.isnan(np.atleast_1d(vals))) and all(np.isnan(np.atleast_1d(vals2)))

        # 1.2/ Check for all NaNs in the raster

        # We create the mask of dilated NaNs
        mask_nan = ~np.isfinite(r.to_nanarray())
        if d != 0:
            mask_nan_dilated = binary_dilation(mask_nan, iterations=d).astype("uint8")
        # (Zero iteration triggers a different behaviour than just "doing nothing" in binary_dilation, we override here)
        else:
            mask_nan_dilated = mask_nan.astype("uint8")
        # Get indices of the related pixels, convert to coordinates
        i, j = np.where(mask_nan_dilated)  # type: ignore
        x, y = r.ij2xy(i, j)
        # And interpolate at those coordinates
        vals = r.interp_at_points(
            (x, y), method=method, force_scipy_function="map_coordinates", nodata_handling=dist, as_array=True
        )
        vals2 = r.interp_at_points(
            (x, y), method=method, force_scipy_function="interpn", nodata_handling=dist, as_array=True
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
        vals = r.interp_at_points(
            (x, y), method=method, force_scipy_function="map_coordinates", nodata_handling=dist, as_array=True
        )
        vals2 = r.interp_at_points(
            (x, y), method=method, force_scipy_function="interpn", nodata_handling=dist, as_array=True
        )

        # Then we fill the NaNs in the raster with a placeholder value of the raster mean
        r_arr = r.to_nanarray()
        r_arr[~np.isfinite(r_arr)] = np.nanmean(r_arr)
        r2 = r.copy(new_array=r_arr)

        # All raster values should be valid now
        assert np.all(np.isfinite(r_arr))

        # Only check accuracy with half-order-up spreading, which covers the complete interpolation kernel
        if dist == "half_order_up":
            # Get the interpolated values
            vals_near = r2.interp_at_points(
                (x, y), method=method, force_scipy_function="map_coordinates", nodata_handling=dist, as_array=True
            )
            vals2_near = r2.interp_at_points(
                (x, y), method=method, force_scipy_function="interpn", nodata_handling=dist, as_array=True
            )

            # Both sets of values should be valid + within a relative tolerance of 0.01%
            assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-4)
            assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-4)

            # Same check for values within one pixel of the exact coordinates
            xoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))
            yoffset = np.random.default_rng(42).uniform(low=-0.5, high=0.5, size=len(x))

            vals = r.interp_at_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="map_coordinates",
                nodata_handling=dist,
                as_array=True,
            )
            vals2 = r.interp_at_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="interpn",
                nodata_handling=dist,
                as_array=True,
            )
            vals_near = r2.interp_at_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="map_coordinates",
                nodata_handling=dist,
                as_array=True,
            )
            vals2_near = r2.interp_at_points(
                (x + xoffset, y + yoffset),
                method=method,
                force_scipy_function="interpn",
                nodata_handling=dist,
                as_array=True,
            )

            # Both sets of values should be exactly the same, without any NaNs
            assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-4)
            assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-4)

            # 2.2/ Repeat the same for all edges of NaNs in the raster
            mask_dilated_plus_one = binary_dilation(mask_nan_dilated, iterations=1).astype(bool)
            mask_edge_dilated = np.logical_and(mask_dilated_plus_one, ~mask_nan_dilated.astype(bool))

            # Get indices of the related pixels, convert to coordinates
            i, j = np.where(mask_edge_dilated)  # type: ignore
            x, y = r.ij2xy(i, j)
            # And interpolate at those coordinates
            vals = r.interp_at_points(
                (x, y), method=method, force_scipy_function="map_coordinates", nodata_handling=dist, as_array=True
            )
            vals2 = r.interp_at_points(
                (x, y), method=method, force_scipy_function="interpn", nodata_handling=dist, as_array=True
            )
            vals_near = r2.interp_at_points(
                (x, y), method=method, force_scipy_function="map_coordinates", nodata_handling=dist, as_array=True
            )
            vals2_near = r2.interp_at_points(
                (x, y), method=method, force_scipy_function="interpn", nodata_handling=dist, as_array=True
            )

            # Both sets of values should be exactly the same, without any NaNs
            assert np.allclose(vals, vals_near, equal_nan=False, rtol=10e-4)
            assert np.allclose(vals2, vals2_near, equal_nan=False, rtol=10e-4)


@pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
class TestInterpolatorKriging:
    """Test module for kriging coefficients, prediction variance and agreement between covariance backends."""

    @pytest.mark.parametrize("error_variance", [0.0, 4.0])
    def test_kriging_variance__observation_error(self, error_variance: float) -> None:
        """Checks that prediction variance includes measurement error when only one observation is available."""

        # A single observation has coefficient one, so its measurement variance contributes without scaling
        model = VariogramModel("gaussian", effective_range=2, partial_sill=1)
        data = LocalData(
            values=np.array([3.0]),
            valid=np.array([True]),
            source_ids=np.array([0]),
            coordinates=np.array([[0.0, 0.0]]),
            target=np.array([1.0, 0.0]),
            error_covariance=np.array([[error_variance]]),
        )
        operator = Kriging(model)

        # The field difference has variance twice its semivariance; independent measurement error adds to it
        assert operator.evaluate(data) == 3
        expected_variance = 2 * float(model.variogram(1)) + error_variance
        assert operator.kriging_variance(data) == pytest.approx(expected_variance)

    def test_kriging__coefficients_match_gstools(self) -> None:
        """Checks that exposed ordinary-kriging weights reproduce GSTools values and variance exactly."""

        # Triangle of observations with interior target
        coordinates = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        values = np.array([1.0, 2.0, 4.0])
        target = np.array([0.4, 0.3])
        local = LocalData(
            values=values,
            valid=np.ones(3, dtype=bool),
            source_ids=np.arange(3),
            coordinates=coordinates,
            target=target,
            distances=np.linalg.norm(coordinates - target, axis=1),
        )
        model = VariogramModel("gaussian", effective_range=2, partial_sill=1)

        # Compare with GSTools ordinary kriging using same model
        operator = Kriging(model)
        converted = operator.variogram.to_gstools(dim=2)
        gstools = pytest.importorskip("gstools")
        reference = gstools.krige.Ordinary(
            converted.model,
            cond_pos=(coordinates[:, 0], coordinates[:, 1]),
            cond_val=values,
            exact=True,
        )
        expected, expected_variance = reference(([target[0]], [target[1]]), store=False)
        assert operator.evaluate(local) == pytest.approx(expected[0], rel=0, abs=1e-14)
        assert operator.kriging_variance(local) == pytest.approx(expected_variance[0], rel=0, abs=1e-14)

        # Check cached coefficients after shifting coordinates and values
        assert len(operator._coefficient_cache) == 1
        shifted = LocalData(
            values=values + 10,
            valid=local.valid,
            source_ids=local.source_ids,
            coordinates=coordinates + 5,
            target=target + 5,
            distances=local.distances,
        )
        assert operator.evaluate(shifted) == pytest.approx(expected[0] + 10, rel=0, abs=1e-14)
        assert len(operator._coefficient_cache) == 1

        # Check cached coefficients after reordering observations
        order = np.array([2, 0, 1])
        assert local.distances is not None
        reordered = LocalData(
            values=values[order],
            valid=local.valid[order],
            source_ids=local.source_ids[order],
            coordinates=coordinates[order],
            target=target,
            distances=local.distances[order],
        )
        assert operator.evaluate(reordered) == pytest.approx(expected[0], rel=0, abs=1e-14)
        assert len(operator._coefficient_cache) == 1

    def test_kriging_variance__one_source_keeps_distance_dependence(self) -> None:
        """Checks that kriging with one source gives it a weight of one but calculates variance from its distance."""

        # Single observation away from target, Gaussian covariance
        source = np.array([[0.0, 0.0]])
        target = np.array([1.0, 0.0])
        local = LocalData(
            values=np.array([3.0]),
            valid=np.ones(1, dtype=bool),
            source_ids=np.array([0]),
            coordinates=source,
            target=target,
            distances=np.array([1.0]),
        )
        operator = Kriging(VariogramModel("gaussian", effective_range=2, partial_sill=1))

        # Compare unit value weight and distance-dependent variance with GSTools
        converted = operator.variogram.to_gstools(dim=2)
        gstools = pytest.importorskip("gstools")
        reference = gstools.krige.Ordinary(
            converted.model,
            cond_pos=(source[:, 0], source[:, 1]),
            cond_val=local.values,
            exact=True,
        )
        expected, expected_variance = reference(([target[0]], [target[1]]), store=False)
        assert operator.evaluate(local) == pytest.approx(expected[0], rel=0, abs=1e-14)
        assert operator.kriging_variance(local) == pytest.approx(expected_variance[0], rel=0, abs=1e-14)

    @pytest.mark.skipif(find_spec("gpytorch") is None or find_spec("torch") is None, reason="Requires GPyTorch")
    def test_kriging__gpytorch_matches_gstools(self) -> None:
        """Checks that the optional GPyTorch covariance solve agrees with GSTools ordinary kriging."""

        # Exponential model with observation noise supported by both backends
        coordinates = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        target = np.array([0.8, 1.1])
        local = LocalData(
            values=np.array([1.0, 3.0, 2.0, 5.0]),
            valid=np.ones(4, dtype=bool),
            source_ids=np.arange(4),
            coordinates=coordinates,
            target=target,
            distances=np.linalg.norm(coordinates - target, axis=1),
        )
        model = VariogramModel("exponential", effective_range=4, partial_sill=2, nugget=0.2)

        # Compare solvers with matching covariance and smoothing settings
        gstools_operator = Kriging(model, backend="gstools", exact=False)
        gpytorch_operator = Kriging(model, backend="gpytorch", exact=False)
        gstools_result = gstools_operator.evaluate(local)
        gpytorch_result = gpytorch_operator.evaluate(local)
        assert gpytorch_result == pytest.approx(gstools_result, rel=1e-10, abs=1e-10)
        assert gpytorch_operator.kriging_variance(local) == pytest.approx(
            gstools_operator.kriging_variance(local), rel=1e-10, abs=1e-10
        )
