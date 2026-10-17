"""Compare public string methods, backend choices and advanced operator objects."""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from affine import Affine

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.operators.interpolator import Kriging, Linear
from geoutils.operators.methods import ResamplingMethod
from geoutils.operators.reducer import Mean, Median, Sum
from geoutils.stats.variography import VariogramModel


class TestResamplingNames:
    """Test module for built-in string interpolation and statistics through the public raster APIs."""

    @pytest.mark.parametrize(
        "resampling",
        [
            "nearest",
            "linear",
            "bilinear",
            "cubic",
            "quintic",
            "slinear",
            "pchip",
            "splinef2d",
            "cubic_convolution",
            "cubic_spline",
            "lanczos",
            "idw",
        ],
    )
    def test_resample_at_points__constant_interpolation(self, resampling: ResamplingMethod) -> None:
        """Checks that interpolation names preserve a constant raster at fractional point coordinates."""

        # Create enough rows and columns for the degree-five spline
        values = np.full((9, 9), 2.0)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 9), 32631)
        point = (np.array([3.8]), np.array([4.5]))

        # Predict inside the raster through the common string interface
        result = raster.resample_at_points(point, resampling=resampling, backend="scipy", as_array=True)

        # Weighted sums and spline solves reproduce the constant within floating point precision
        np.testing.assert_allclose(result, [2.0], rtol=0, atol=1e-12)

    @pytest.mark.parametrize(
        "resampling, statistic",
        [
            ("mean", np.mean),
            ("average", np.mean),
            ("sum", np.sum),
            ("median", np.median),
            ("minimum", np.min),
            ("maximum", np.max),
            ("range", np.ptp),
            ("stdev", np.std),
            ("std", np.std),
            ("rms", lambda values: np.sqrt(np.mean(values**2))),
        ],
    )
    def test_resample_at_points__window_statistics(
        self, resampling: ResamplingMethod, statistic: Callable[[Any], Any]
    ) -> None:
        """Checks that statistic names summarize the requested window with the corresponding NumPy result."""

        # Create a known central 3 x 3 window in a finite elevation raster
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        point = raster.ij2xy(2, 2)
        expected = statistic(values[1:4, 1:4])

        # Summarize the window without constructing a Reducer
        result = raster.resample_at_points(point, resampling=resampling, backend="scipy", window=3, as_array=True)

        # Floating point reductions can differ in their final rounding
        np.testing.assert_allclose(result, [expected], rtol=0, atol=1e-12)

    def test_resample_at_points__quantile_options(self) -> None:
        """Checks that quantile settings and named quartiles reproduce NumPy's quantiles."""

        # Create a central window with distinct values so quartile interpolation is visible
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        point = raster.ij2xy(2, 2)

        # Calculate a supplied quantile and the upper quartile alias
        quantile = raster.resample_at_points(point, resampling="quantile", resampling_kwargs={"q": 0.75}, as_array=True)
        upper_quartile = raster.resample_at_points(point, resampling="q3", as_array=True)

        # Both calls select the same value from the central 3 x 3 window
        expected = [np.quantile(values[1:4, 1:4], 0.75)]
        np.testing.assert_array_equal(quantile, expected)
        np.testing.assert_array_equal(upper_quartile, expected)

    def test_at_points__legacy_aliases_and_advanced_objects(self) -> None:
        """Checks that existing keywords and advanced objects produce the same values as resampling strings."""

        # Create a plane and two fractional targets for a known linear prediction
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        points = np.array([1.2, 2.3]), np.array([3.2, 2.4])

        # Compare the new keyword, the legacy keyword and an advanced interpolator
        expected = raster.interp_at_points(points, method="linear", as_array=True)
        string_result = raster.interp_at_points(points, resampling="linear", backend="scipy", as_array=True)
        advanced_result = raster.resample_at_points(points, resampling=Linear(), backend="scipy", as_array=True)

        # All forms use the same interpolation weights
        np.testing.assert_array_equal(string_result, expected)
        np.testing.assert_array_equal(advanced_result, expected)


class TestResamplingBackends:
    """Test module for named reprojection, filtering, gridding and gap-filling implementations."""

    @pytest.mark.parametrize("backend", ["gdal", "scipy"])
    def test_reproject__named_backend(self, backend: Literal["gdal", "scipy"]) -> None:
        """Checks that a named backend reproduces its established mean reprojection calculation."""

        # Create a raster whose coarser cells contain different source values
        values = np.arange(36, dtype=float).reshape(6, 6)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 6), 32631)
        method = rio.enums.Resampling.average if backend == "gdal" else Mean()

        # Compare the string call with the existing enum or advanced operator
        expected = raster.reproject(res=2, resampling=method)
        result = raster.reproject(res=2, resampling="mean", backend=backend)

        # Each backend retains its own established area calculation
        assert result is not None
        assert expected is not None
        np.testing.assert_array_equal(result.to_nanarray(), expected.to_nanarray())

    @pytest.mark.parametrize("backend", ["scipy", "numba"])
    def test_filter__named_backend_and_fractional_window(self, backend: Literal["scipy", "numba"]) -> None:
        """Checks that named filters select the same backend and fractional window as advanced reducers."""

        # Create a finite raster and a circular window cutting through its edge cells
        if backend == "numba":
            import_optional("numba")
        values = np.arange(49, dtype=float).reshape(7, 7)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 7), 32631)
        options: dict[str, Any] = {"size": 5, "kernel_shape": "circular", "coverage": "fractional"}

        # Compare the named mean with the advanced reducer and legacy engine keyword
        expected = raster.filter(Mean(), engine=backend, **options)
        result = raster.filter(method="mean", backend=backend, **options)

        # Both forms use the same coverage weights and preserve the raster grid
        np.testing.assert_array_equal(result.to_nanarray(), expected.to_nanarray())
        assert result.transform == raster.transform

    @pytest.mark.parametrize("resampling, operator", [("median", Median()), ("sum", Sum())])
    def test_grid__additional_statistic_names(self, resampling: ResamplingMethod, operator: Any) -> None:
        """Checks that statistic names grid points with the same values as advanced reducer objects."""

        # Convert a finite raster to points so both calls share exact locations and measurements
        values = np.arange(25, dtype=float).reshape(5, 5)
        reference = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        points = reference.to_pointcloud()

        # Grid with the new name and the previously required reducer object
        expected = points.grid(ref=reference, resampling=operator, dist_nodata_pixel=2, nodata_handling="ignore")
        result = points.grid(
            ref=reference, resampling=resampling, backend="scipy", dist_nodata_pixel=2, nodata_handling="ignore"
        )

        # The selected point neighborhoods and their statistics are unchanged
        np.testing.assert_array_equal(result.to_nanarray(), expected.to_nanarray())

    def test_filter__point_dataset_keywords(self) -> None:
        """Checks that Dataset point filters accept the same method and backend keywords as DataArrays."""

        # Create two measurement variables on the same point coordinates
        raster = gu.Raster.from_array(np.arange(25, dtype=float).reshape(5, 5), Affine(1, 0, 0, 0, -1, 5), 32631)
        points = raster.to_pointcloud().to_xarray()
        dataset = xr.Dataset({"height": points, "offset_height": points + 10})

        # Filter both variables through the Dataset dispatch path
        result = dataset.pc.filter(method="mean", backend="scipy", radius=2)
        expected = points.pc.filter(method="mean", radius=2)

        # Both variables receive the same neighborhood mean with their original offset
        np.testing.assert_array_equal(result.height.data, expected.data)
        # Averaging offset values can round differently from adding the offset after averaging
        np.testing.assert_allclose(result.offset_height.data, expected.data + 10, rtol=0, atol=1e-14)

    def test_fill_nodata__resampling_and_backend(self) -> None:
        """Checks that the common keywords reproduce the existing GDAL gap-filling options."""

        # Create a missing center in an otherwise finite raster
        values = np.arange(25, dtype=float).reshape(5, 5)
        values[2, 2] = np.nan
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631, nodata=-9999)

        # Compare the new IDW name with the legacy GDAL interpolation name
        expected = raster.fill_nodata(interpolation="inv_dist", max_search_distance=2)
        result = raster.fill_nodata(resampling="idw", backend="gdal", max_search_distance=2)

        # Gap filling uses the same directional GDAL search and leaves finite cells unchanged
        np.testing.assert_array_equal(result.to_nanarray(), expected.to_nanarray())


class TestNeighborhoodKeywords:
    """Test module for deprecated neighborhood keywords and their warnings."""

    @pytest.mark.parametrize("options", [{"size": 3}, {"kernel_shape": "circular"}])
    def test_filter__deprecated_window_keywords(self, options: dict[str, Any]) -> None:
        """Checks that filter() runs with deprecated window keywords and raises a warning."""

        # Create a small constant raster
        raster = gu.Raster.from_array(np.ones((3, 3)), Affine(1, 0, 0, 0, -1, 3), 32631)

        # Run the filter with each old keyword and check its warning
        with pytest.warns(DeprecationWarning, match="deprecated"):
            raster.filter(method="mean", **options)

    def test_grid__deprecated_radius_keyword(self) -> None:
        """Checks that grid() runs with dist_nodata_pixel and raises a warning."""

        # Create a small raster and matching points
        reference = gu.Raster.from_array(np.ones((3, 3)), Affine(1, 0, 0, 0, -1, 3), 32631)
        points = reference.to_pointcloud()

        # Run gridding with the old distance keyword and check its warning
        with pytest.warns(DeprecationWarning, match="dist_nodata_pixel.*deprecated.*radius"):
            points.grid(ref=reference, resampling="nearest", dist_nodata_pixel=1)


class TestResamplingKriging:
    """Test module for named kriging solvers and advanced covariance model options."""

    @pytest.mark.parametrize("backend", ["auto", "gstools", "gpytorch"])
    def test_resample_at_points__named_kriging_backend(self, backend: Literal["auto", "gstools", "gpytorch"]) -> None:
        """Checks that a named kriging solver agrees exactly with its advanced Kriging object."""

        # Create nearby observations and a fitted model for the requested solver
        solver: Literal["gstools", "gpytorch"] = "gstools" if backend == "auto" else backend
        if find_spec(solver) is None:
            pytest.skip(f"Requires {solver}")
        import_optional(solver)
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        model = VariogramModel("exponential", effective_range=3, partial_sill=1)
        point = (np.array([2.2]), np.array([2.8]))

        # Compare a string and model settings with the corresponding configured object
        expected = raster.resample_at_points(point, Kriging(model, backend=solver), as_array=True)
        result = raster.resample_at_points(
            point, resampling="kriging", backend=backend, resampling_kwargs={"model": model}, as_array=True
        )

        # Both calls use identical observations, covariance and solver settings
        np.testing.assert_array_equal(result, expected)

    def test_resample_at_points__advanced_backend_override(self) -> None:
        """Checks that choosing a solver leaves the supplied Kriging object's configuration and cache unchanged."""

        # Create a GSTools operator and populate its coefficients with a first prediction
        for dependency in ("gstools", "gpytorch"):
            if find_spec(dependency) is None:
                pytest.skip(f"Requires {dependency}")
            import_optional(dependency)
        raster = gu.Raster.from_array(np.arange(25, dtype=float).reshape(5, 5), Affine(1, 0, 0, 0, -1, 5), 32631)
        model = VariogramModel("exponential", effective_range=3, partial_sill=1)
        operator = Kriging(model, backend="gstools")
        point = (np.array([2.2]), np.array([2.8]))
        original_prediction = raster.resample_at_points(point, resampling=operator, as_array=True)
        original_cache = operator._coefficient_cache.copy()
        assert original_cache

        # Override the solver for this call and compare with a separately configured GPyTorch calculation
        result = raster.resample_at_points(point, resampling=operator, backend="gpytorch", as_array=True)
        expected = raster.resample_at_points(point, resampling=Kriging(model, backend="gpytorch"), as_array=True)

        # The override changes neither the supplied solver nor its cached GSTools calculation
        np.testing.assert_array_equal(result, expected)
        assert operator.backend == "gstools"
        assert operator._coefficient_cache.keys() == original_cache.keys()
        for key in original_cache:
            np.testing.assert_array_equal(operator._coefficient_cache[key][0], original_cache[key][0])
            assert operator._coefficient_cache[key][1] == original_cache[key][1]
        np.testing.assert_array_equal(
            raster.resample_at_points(point, resampling=operator, as_array=True), original_prediction
        )


class TestResamplingNamesChunked:
    """Test module for lazy named resampling and exact agreement with eager window reductions."""

    def test_resample_at_points__loading_laziness(self) -> None:
        """Checks that string window reductions stay lazy and match eager results across shorter final chunks."""

        # Split five rows and columns into chunks of two, leaving a shorter final chunk on each axis
        if find_spec("dask") is None:
            pytest.skip("Requires Dask")
        dask_array = import_optional("dask").array
        values = np.arange(25, dtype=float).reshape(5, 5)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631)
        lazy = raster.to_xarray().chunk({"x": 2, "y": 2})
        points = raster.ij2xy(np.array([1, 2, 3]), np.array([1, 2, 3]))

        # Request sums whose windows cross the chunk boundaries
        result = lazy.rst.resample_at_points(points, resampling="sum", backend="scipy", window=3, as_array=True)
        expected = raster.resample_at_points(points, resampling="sum", backend="scipy", window=3, as_array=True)

        # Input and output remain lazy; computing the result reproduces each complete eager window
        assert isinstance(lazy.data, dask_array.Array)
        assert isinstance(result, dask_array.Array)
        np.testing.assert_array_equal(result.compute(), expected)
        assert isinstance(lazy.data, dask_array.Array)


class TestResamplingErrors:
    """Test module for conflicting keywords, unavailable backends and incomplete method settings."""

    @pytest.mark.parametrize(
        "options, error",
        [
            ({"method": "linear", "resampling": "nearest"}, "Pass only one"),
            ({"resampling": "mean", "backend": "gdal"}, "unavailable"),
            ({"resampling": "quantile"}, "requires q"),
            ({"resampling": "kriging"}, "requires a fitted model"),
            ({"resampling": "kriging", "backend": "scipy"}, "Kriging requires backend"),
            ({"resampling": "nearest", "backend": "gpytorch"}, "requires resampling='kriging'"),
        ],
    )
    def test_resample_at_points__error_invalid_selection(self, options: dict[str, Any], error: str) -> None:
        """Checks an error is raised for conflicting choices, unavailable backends or missing method settings."""

        # Create a finite raster so errors concern method selection rather than missing observations
        raster = gu.Raster.from_array(np.ones((5, 5)), Affine(1, 0, 0, 0, -1, 5), 32631)
        point = raster.ij2xy(2, 2)

        # Request the invalid public keyword combination
        with pytest.raises(ValueError, match=error):
            raster.resample_at_points(point, as_array=True, **options)

    @pytest.mark.parametrize("resampling", ["cubic_convolution", "cubic_spline", "lanczos", "quintic"])
    def test_grid__error_regular_only_interpolation(self, resampling: ResamplingMethod) -> None:
        """Checks an error is raised for raster-only interpolation methods when gridding irregular points."""

        # Create measurements with a raster reference to avoid unrelated grid-definition errors
        reference = gu.Raster.from_array(np.ones((5, 5)), Affine(1, 0, 0, 0, -1, 5), 32631)
        points = reference.to_pointcloud()

        # Request a method whose convolution or spline requires a regular source grid
        with pytest.raises(ValueError, match="regular"):
            points.grid(ref=reference, resampling=resampling, backend="scipy")

    def test_filter__error_conflicting_backend_alias(self) -> None:
        """Checks an error is raised when backend and the legacy engine select different calculations."""

        # Create a finite raster so method selection is the only invalid input
        raster = gu.Raster.from_array(np.ones((5, 5)), Affine(1, 0, 0, 0, -1, 5), 32631)

        # Select incompatible implementations through the new and legacy keywords
        with pytest.raises(ValueError, match="Pass only one"):
            raster.filter(method="mean", backend="scipy", engine="numba", size=3)
