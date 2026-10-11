"""Tests for analytical and numerical uncertainty propagation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import rasterio as rio

import geoutils as gu
from geoutils.operators import Interpolator, LinearCoefficients, LocalData, Reducer
from geoutils.operators.neighbours import GridNeighbours, PointNeighbours
from geoutils.operators.reducer import Mean
from geoutils.stats.variography import VariogramModel
from geoutils.uncertainty.propagation import simulate
from tests.operator_helpers import LocalMeanInterpolator, NoSupportReducer


class NonlinearMeanSquare(Reducer):
    """Nonlinear function (square mean) to test uncertainty propagation for nonlinear cases."""

    def reduce(self, data: LocalData) -> float:

        return float(np.mean(data.values) ** 2)


class CountingMean(Reducer):
    """Function to count calls to reductions to check whether a mean is recalculated during uncertainty propagation."""

    def __init__(self) -> None:

        self.calls = 0

    def reduce(self, data: LocalData) -> float:

        self.calls += 1
        return float(np.mean(data.values))


class FirstValue(Interpolator):
    default_nodata_propagation = "propagate"

    def coefficients(self, data: LocalData) -> LinearCoefficients:

        weights = np.zeros(len(data.values), dtype=float)
        weights[0] = 1
        return LinearCoefficients(weights)


def _local(values: list[float], source_ids: list[str], coordinates: list[list[float]]) -> LocalData:
    """Helper to create local data for small propagation calculations."""

    return LocalData(
        values=np.asarray(values, dtype=float),
        valid=np.ones(len(values), dtype=bool),
        source_ids=np.asarray(source_ids),
        coordinates=np.asarray(coordinates, dtype=float),
    )


class TestAnalyticalOperatorPropagation:
    """Test module for analytical uncertainty propagation for operators."""

    def test_propagate__id_independence(self) -> None:
        """Checks that repeated IDs are fully correlated, and distinct IDs independent."""

        # We create coincident observations with distinct/repeat IDs to test dependence during propagation
        first = _local([10, 20], ["a", "b"], [[0, 0], [0, 0]])
        second = _local([10, 10], ["a", "a"], [[0, 0], [0, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # We propagate uncertainty through a mean operation
        # Distinct IDs have separate variance
        # Repeated IDs are correlated
        summary = gu.uncertainty.propagate(Mean(), [first, second], structure)
        np.testing.assert_allclose(summary.estimate, [15, 10])  # Means of 15 and 10
        np.testing.assert_allclose(summary.variance, [2, 4])  # Repeat = 2, Distinct = 4

    def test_propagate__magnitude_id(self) -> None:
        """Checks that predictors assign each source its own grouped error magnitude before averaging."""

        # We create magnitudes grouped by slope + source values in B/A order
        intervals = pd.IntervalIndex.from_breaks([0, 1, 2], name="slope")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        statistics = pd.DataFrame([[1.0, 10], [2.0, 10]], index=intervals, columns=columns)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        data = _local([10, 20], ["b", "a"], [[0, 0], [1, 0]])

        # Inverse-variance weights: 4/5 for B, 1/5 for A
        summary = gu.uncertainty.propagate(Mean(), data, structure, predictors={"slope": {"a": 1.5, "b": 0.5}})
        assert summary.estimate == pytest.approx(12)
        assert summary.variance == pytest.approx(0.8)

    def test_propagate__array_id_order(self) -> None:
        """Checks that array predictors use numeric IDs (not order)."""

        # We create grouped errors in ascending ID order, but values in reverse order
        intervals = pd.IntervalIndex.from_breaks([0, 1, 2, 3], name="slope")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        statistics = pd.DataFrame([[1.0, 10], [2.0, 10], [3.0, 10]], index=intervals, columns=columns)
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("measurement", gu.ErrorMagnitude.variable_from_grouped_stats(statistics))]
        )
        data = LocalData(
            values=np.array([10.0, 20.0, 30.0]),
            valid=np.ones(3, dtype=bool),
            source_ids=np.array([10, 2, 1]),
        )

        # Check ID 10 uses last predictor (numeric order), despite first row position
        summary = gu.uncertainty.propagate(FirstValue(), data, structure, predictors={"slope": [0.5, 1.5, 2.5]})
        assert summary.estimate == 10
        assert summary.std == pytest.approx(3)

    def test_propagate__zero_coef(self) -> None:
        """Checks edge case that an invalid input with 0 coef does not affect the result."""

        # We create local data with a NaN value that is masked
        data = LocalData(
            values=np.array([3.0, np.nan]),
            valid=np.array([True, False]),
            source_ids=np.array(["used", "unused"]),
            coordinates=np.array([[0.0, 0.0], [1.0, 0.0]]),
        )
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])

        # Check it does not affect output
        summary = gu.uncertainty.propagate(FirstValue(), data, structure)
        assert summary.estimate == pytest.approx(3)
        assert summary.std == pytest.approx(0.5)

    def test_propagate__undefined_uncertainty(self) -> None:
        """Checks edge case that a fully-NaN input has no propagated results."""

        # Create NaN input
        data = LocalData(values=np.array([np.nan]), valid=np.array([False]), source_ids=np.array(["missing"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check NaN estimate and moments
        summary = gu.uncertainty.propagate(Mean(), data, structure)
        assert np.isnan(summary.estimate)
        assert np.isnan(summary.mean)
        assert np.isnan(summary.std)

    @pytest.mark.parametrize(
        "at, selected",
        [("all", ["left", "right"]), (["right"], ["right"])],
    )
    def test_propagate__selects_analytical_quantiles(self, at: Literal["all"] | list[str], selected: list[str]) -> None:
        """Checks that outputs determine the analytical quantile columns."""

        # We create two data that use "b"
        first = _local([1, 2], ["a", "b"], [[0, 0], [1, 0]])
        second = _local([2, 3], ["b", "c"], [[1, 0], [2, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Propagate
        summary = gu.uncertainty.propagate(
            Mean(),
            [first, second],
            structure,
            output_labels=["left", "right"],
            at=at,
            quantiles=(0.5,),
        )

        # Each mean has variance 1/2 and median equal to its original value
        labels = pd.Index(["left", "right"], name="output")
        np.testing.assert_allclose(summary.variance, [0.5, 0.5])
        assert summary.quantiles is not None
        expected_medians = pd.Series([np.mean(first.values), np.mean(second.values)], index=labels)
        np.testing.assert_array_equal(summary.quantiles.loc[0.5, selected], expected_medians.loc[selected])

    def test_propagate__error_return_samples(self) -> None:
        """Checks an error is raised when return_samples is used with analytical."""

        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="return_samples cannot be used with analytical"):
            gu.uncertainty.propagate(Mean(), data, structure, method="analytical", return_samples=True)

    def test_propagate__error_invalid_nominal_estimate(self) -> None:
        """Checks an error is raised for more "nominal_estimate" than targets."""

        # Two observations in one group (one output)
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Passing 2 nominal estimates should raise an error
        with pytest.raises(ValueError, match="one value per LocalData target"):
            gu.uncertainty.propagate(Mean(), data, structure, nominal_estimate=[2, 3])


class TestNumericalOperatorPropagation:
    """Test module for numerical uncertainty propagation for operators."""

    def test_propagate__nonlinear_numerical_fallback(self) -> None:
        """Checks that propagate() simulates the mean and spread of a nonlinear result."""

        # We create small independent errors for a nonlinear mean-square calculation
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])
        summary = gu.uncertainty.propagate(
            NonlinearMeanSquare(),
            data,
            structure,
            n_samples=2_000,
            random_state=7,
        )

        # Expected squared mean: 2**2 + Var(mean) = 4 + 0.2**2 / 2
        assert summary.method == "numerical"
        assert summary.estimate == pytest.approx(4)
        assert summary.mean == pytest.approx(4.02, abs=0.04)

    def test_propagate__auto_draws(self) -> None:
        """Checks that "auto" propagation uses "numerical" when return_sample=True."""

        # Mean with exact coefficients (analytical by default)
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])

        # Compare automatic/explicit numerical draws with same seed
        automatic = gu.uncertainty.propagate(Mean(), data, structure, n_samples=4, random_state=6, return_samples=True)
        numerical = gu.uncertainty.propagate(
            Mean(), data, structure, method="numerical", n_samples=4, random_state=6, return_samples=True
        )
        assert automatic.method == "numerical"
        assert automatic.samples is not None and numerical.samples is not None
        np.testing.assert_array_equal(automatic.samples, numerical.samples)

    def test_propagate__nominal(self) -> None:
        """Checks that a nominal estimate is properly used."""

        # Count operator calls with precomputed mean
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        operator = CountingMean()

        # Should sue the nominal estimate of 2, calling on 3 times
        summary = gu.uncertainty.propagate(
            operator, data, structure, method="numerical", nominal_estimate=2, n_samples=3, random_state=6
        )
        assert summary.estimate == 2
        assert operator.calls == 3

    def test_propagate__masked(self) -> None:
        """Checks that masked values are all excluded from errors and simulations."""

        # We include a mask value (that is finite)
        values = np.ma.array([2.0, 100.0], mask=[False, True])
        data = LocalData(
            values=values,
            valid=np.ones(2, dtype=bool),
            source_ids=np.asarray(["used", "masked"]),
        )
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # We check identical draws with zero error for masked (square of unmasked value = 2)
        summary = gu.uncertainty.propagate(
            NonlinearMeanSquare(), data, errors, method="numerical", n_samples=4, return_samples=True
        )
        assert summary.estimate == 4
        assert summary.samples is not None
        np.testing.assert_array_equal(summary.samples.to_numpy(), np.full((4, 1), 4.0))

    def test_propagate__order_reproducibility(self) -> None:
        """Checks that reversing the input order does not change errors generated with the same seed."""

        # Overlapping inputs passed with reverse order
        first = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        second = _local([3, 5], ["b", "c"], [[1, 0], [2, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        forward = gu.uncertainty.propagate(
            NonlinearMeanSquare(),
            [first, second],
            structure,
            method="numerical",
            n_samples=20,
            random_state=9,
            return_samples=True,
            output_labels=["first", "second"],
        )
        reversed_order = gu.uncertainty.propagate(
            NonlinearMeanSquare(),
            [second, first],
            structure,
            method="numerical",
            n_samples=20,
            random_state=9,
            return_samples=True,
            output_labels=["second", "first"],
        )

        # Check exact matching draws per target after reordering
        assert forward.samples is not None and reversed_order.samples is not None
        np.testing.assert_array_equal(forward.samples["first"], reversed_order.samples["first"])
        np.testing.assert_array_equal(forward.samples["second"], reversed_order.samples["second"])

    def test_propagate__error_samples_invalid(self) -> None:
        """Checks an error is raised when sampling with no valid source."""

        # A source with no valid observation to sample
        data = LocalData(values=np.array([np.nan]), valid=np.array([False]), source_ids=np.array(["missing"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="at least one valid source"):
            gu.uncertainty.propagate(Mean(), data, structure, return_samples=True)


class TestSimulation:
    """Test module for simulate(): numerical simulation with random draws."""

    def test_simulate__finite_draws(self) -> None:
        """Checks that output use only finite draws."""

        # We define three draws, with the second output missing from the middle draw
        draws = iter((np.array([1.0, 2.0]), np.array([3.0, np.nan]), np.array([5.0, 6.0])))
        selection = pd.DataFrame({"flat_index": [0, 1], "estimate": [0.0, 0.0]}, index=["first", "second"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Simulate for both outputs
        summary = simulate(
            estimate=np.zeros(2),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: draw,
            error_structure=structure,
            selection=selection,
            n_samples=3,
            return_samples=True,
            quantiles=(0.5,),
        )

        # The first output should have three finite draws and the second has two
        np.testing.assert_array_equal(summary.n_valid, [3, 2])
        np.testing.assert_allclose(summary.mean, [3, 4])
        np.testing.assert_allclose(summary.std, [2, np.sqrt(8)])
        assert summary.samples is not None
        expected_samples = np.array([[1, 2], [3, np.nan], [5, 6]])
        np.testing.assert_allclose(summary.samples.to_numpy(), expected_samples, equal_nan=True)
        assert summary.quantiles is not None
        np.testing.assert_allclose(summary.quantiles.loc[0.5], [3, 4])

    def test_simulate__circular(self) -> None:
        """Checks mean/std behaviour for circular variables such as angles."""

        # We create to draws with angles 359 and 1 deg (2 deg apart modulo 360)
        draws = iter((359.0, 1.0))
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["angle"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # We simulate a quantile, passing the circularity
        summary = simulate(
            estimate=np.array(0.0),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: np.asarray(draw),
            error_structure=structure,
            selection=selection,
            n_samples=2,
            output={"circular_period": 360.0},
            return_samples=True,
            quantiles=(0.5,),
        )

        # Circular mean should be 0, STD should be roughly 1 (with some trigonometrical considerations)
        assert summary.mean % 360 == pytest.approx(0)
        expected_resultant = np.cos(np.deg2rad(1))
        assert summary.resultant_length == pytest.approx(expected_resultant)
        assert summary.std == pytest.approx(180 / np.pi * np.sqrt(-2 * np.log(expected_resultant)))
        assert summary.quantiles is not None
        assert summary.quantiles.loc[0.5, "angle"] == pytest.approx(0)
        assert summary.samples is not None
        np.testing.assert_array_equal(summary.samples["angle"], [359, 1])

    def test_simulate__warns_failed_draw(self) -> None:
        """Checks that a warning mentions a failed calculation, skipping it to keep only successful draws."""

        # We input a negative draw to make the calculation fail between two finite results
        draws = iter((1.0, -1.0, 3.0))
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        def calculate(draw: float) -> float:
            """We create a calculation that will raise error on the negative draw only."""

            if draw < 0:
                raise ValueError("negative draw")
            return draw

        # Check warning is raised
        with pytest.warns(UserWarning, match="Simulation 2 of 3 failed"):
            summary = simulate(
                estimate=np.array(0.0),
                draw_error=lambda _rng: next(draws),
                calculate=calculate,
                error_structure=structure,
                selection=selection,
                n_samples=3,
                on_error="warn",
                return_samples=True,
            )
        # Successful draws 1/3 have mean 2 and STD of sqrt(2)
        assert summary.n_success == 2
        assert summary.failures == {2: "negative draw"}
        assert summary.mean == pytest.approx(2)
        assert summary.std == pytest.approx(np.sqrt(2))
        assert summary.samples is not None
        np.testing.assert_allclose(summary.samples["value"], [1, np.nan, 3], equal_nan=True)

    def test_simulate__nofinite_draws(self) -> None:
        """Checks that a missing output has an NaN mean and zero valid count."""

        # We create a second output always with NaN values
        draws = iter((np.array([1.0, np.nan]), np.array([3.0, np.nan])))
        selection = pd.DataFrame({"flat_index": [1], "estimate": [np.nan]}, index=["missing"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # We simulate
        summary = simulate(
            estimate=np.array([0.0, np.nan]),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: draw,
            error_structure=structure,
            selection=selection,
            n_samples=2,
        )
        # Check finite first output and missing second output
        np.testing.assert_allclose(summary.mean, [2, np.nan], equal_nan=True)
        np.testing.assert_array_equal(summary.n_valid, [2, 0])


class TestSpatialPropagation:
    """Test module for propagation support through spatial function such as grid, reproject, etc."""

    @pytest.mark.parametrize("engine", ["scipy", "numba"])
    @pytest.mark.parametrize("method", ["nearest", "idw"])
    def test_propagate__grid(self, engine: str, method: str) -> None:
        """Checks propagation with grid()."""

        # We define the error structure and inputs
        if engine == "numba":
            pytest.importorskip("numba")
        points = gu.PointCloud.from_xyz([0.0, 4.0], [1.0, 0.0], [10.0, 20.0], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 0, 10, 1), crs=32631)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        kwargs = {"ref": reference, "resampling": method, "dist_nodata_pixel": 2, "engine": engine}

        # We grid
        expected = points.grid(**kwargs)
        result = points.grid(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(points.grid, error_structure=errors, operation_kwargs=kwargs)

        # We keep the same main output
        assert np.allclose(result.to_nanarray(), expected.to_nanarray(), equal_nan=True)
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), result.to_nanarray())
        assert np.allclose(summary.mean.to_nanarray(), result.to_nanarray(), equal_nan=True)

        # We compute expected variance: target is (0, 0), nearest selects one point
        # and IDW uses inverse squared distances
        source_coordinates = np.array([[0.0, 1.0], [4.0, 0.0]])
        distances = np.linalg.norm(source_coordinates, axis=1)
        if method == "nearest":
            weights = np.zeros(2)
            weights[np.argmin(distances)] = 1
        else:
            weights = distances**-2
            weights /= weights.sum()
        covariance = errors.to_covariance_matrix(source_coordinates)
        expected_variance = weights @ covariance @ weights

        # We check both output variance/STD are as expected
        assert summary.variance.to_nanarray()[0, 0] == pytest.approx(expected_variance)
        assert summary.std.to_nanarray()[0, 0] == pytest.approx(np.sqrt(expected_variance))

    def test_propagate__resample_at_points(self) -> None:
        """Checks that resample at points propagates uncertainty properly."""

        pytest.importorskip("skgstat")

        # We define the error structure and inputs
        raster = gu.Raster.from_array(np.array([[10.0, 20.0]]), rio.transform.from_origin(0, 1, 1, 1), crs=32631)
        correlation = VariogramModel("spherical", effective_range=5, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])
        operator = Mean(GridNeighbours(offsets=((0, 0), (0, 1))))
        kwargs = {"points": ([0.5], [0.5]), "method": operator, "as_array": True}

        # We compare direct/propagated results with generalized least squares
        result = raster.resample_at_points(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(raster.resample_at_points, error_structure=errors, operation_kwargs=kwargs)
        covariance = errors.to_covariance_matrix(np.array([[0.5, 0.5], [1.5, 0.5]]))
        weights = np.linalg.solve(covariance, np.ones(2))
        weights /= weights.sum()

        # We compare propagated covariance with what we expect
        expected_variance = weights @ covariance @ weights
        assert result == pytest.approx(weights @ [10, 20])
        assert summary.estimate == result
        assert summary.variance == pytest.approx(expected_variance)
        assert summary.std == pytest.approx(np.sqrt(expected_variance))
        assert summary.variance > 2

    @pytest.mark.parametrize("operation", ["grid", "resample_at_points", "reproject"])
    def test_propagate__dtype(self, operation: Literal["grid", "resample_at_points", "reproject"]) -> None:
        """Checks that a fractional error magnitude is returned as floating when source raster is integer-type."""

        # We provide four independent errors of magnitude one (will give an STD on 1/2 when averaged)
        raster = gu.Raster.from_array(
            np.array([[2, 4], [6, 8]], dtype=np.int16), rio.transform.from_origin(0, 2, 1, 1), crs=32631, nodata=-9999
        )
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 2, 2, 2), crs=32631)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Average the same four integer values with each spatial API
        if operation == "grid":
            points = gu.PointCloud.from_xyz(
                [0.5, 1.5, 0.5, 1.5], [1.5, 1.5, 0.5, 0.5], raster.data.ravel(), crs=raster.crs
            )
            assert np.issubdtype(points.data.dtype, np.integer)
            spatial_method: Callable[..., Any] = points.grid
            kwargs: dict[str, Any] = {"ref": reference, "resampling": Mean(PointNeighbours(k=4))}
        elif operation == "resample_at_points":
            neighborhood = GridNeighbours(offsets=((0, 0), (0, 1), (1, 0), (1, 1)))
            spatial_method = raster.resample_at_points
            kwargs = {"points": ([0.5], [1.5]), "method": Mean(neighborhood), "as_array": True}
        else:
            spatial_method = raster.reproject
            kwargs = {"ref": reference, "resampling": Mean()}

        # Propagate errors through the selected method
        summary = gu.uncertainty.propagate(spatial_method, error_structure=errors, operation_kwargs=kwargs)

        # We check output STD is floating-type despite the input
        estimate = summary.estimate.to_nanarray() if isinstance(summary.estimate, gu.Raster) else summary.estimate
        std = summary.std.to_nanarray() if isinstance(summary.std, gu.Raster) else summary.std
        assert np.asarray(estimate).item() == 5
        assert np.issubdtype(np.asarray(std).dtype, np.floating)
        assert np.asarray(std).item() == 0.5

    def test_propagate__reproject_bilinear(self) -> None:
        """Checks that bilinear interpolation uses geometric weights with spatially correlated errors."""

        pytest.importorskip("skgstat")

        # We define a target halfway between four cells (bilinear weights 1/4 each)
        raster = gu.Raster.from_array(
            np.array([[0.0, 4.0], [8.0, 12.0]]), rio.transform.from_origin(0, 2, 1, 1), crs=32631
        )
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.5, 1.5, 1, 1), crs=32631)
        correlation = VariogramModel("spherical", effective_range=5, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])
        kwargs = {"ref": reference, "resampling": "bilinear"}

        # We reproject and compute the expected propagated error
        result = raster.reproject(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(raster.reproject, error_structure=errors, operation_kwargs=kwargs)
        centers = np.array([[0.5, 1.5], [1.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
        covariance = errors.to_covariance_matrix(centers)
        weights = np.full(4, 0.25)
        expected_variance = weights @ covariance @ weights

        # We check the propagation is correct
        assert result.to_nanarray()[0, 0] == 6
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), result.to_nanarray())
        assert summary.variance.to_nanarray()[0, 0] == pytest.approx(expected_variance)
        assert summary.std.to_nanarray()[0, 0] == pytest.approx(np.sqrt(expected_variance))

    @pytest.mark.parametrize("operation", ["grid", "resample_at_points", "reproject"])
    @pytest.mark.parametrize(
        "operator_type, statistic",
        [(LocalMeanInterpolator, np.mean), (NoSupportReducer, np.sum)],
        ids=["interpolator", "reducer"],
    )
    @pytest.mark.parametrize("return_samples", [False, True], ids=["moments", "samples"])
    def test_propagate__custom_operator(
        self,
        operation: Literal["grid", "resample_at_points", "reproject"],
        operator_type: type[Interpolator] | type[Reducer],
        statistic: Callable[[Any], Any],
        return_samples: bool,
    ) -> None:
        """Checks that custom operators return moments and samples matching the same source draws."""

        # We define 9 values to file a 3x3 window around (1.5, 1.5)
        values = np.arange(9, dtype=float).reshape(3, 3)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 3, 1, 1), crs=32632)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 3, 3, 3), crs=32632)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # We define the neighborhoods to use all these values
        if operation == "grid":
            x, y = np.meshgrid(np.arange(3) + 0.5, 2.5 - np.arange(3))
            points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values.ravel(), crs=32632)
            operator = operator_type(neighborhood=PointNeighbours(k=9))
            spatial_method = points.grid
            kwargs: dict[str, Any] = {"ref": reference, "resampling": operator, "dist_nodata_pixel": 0}
        else:
            operator = operator_type(neighborhood=GridNeighbours(size=3))
            spatial_method = getattr(raster, operation)
            if operation == "resample_at_points":
                kwargs = {"points": ([1.5], [1.5]), "method": operator, "as_array": True}
            else:
                kwargs = {"ref": reference, "resampling": operator}

        # We perform propagation using numerical draws
        summary = gu.uncertainty.propagate(
            spatial_method,
            error_structure=errors,
            operation_kwargs=kwargs,
            n_samples=8,
            random_state=6,
            return_samples=return_samples,
        )

        # We independently draw the samples with the same random state, and estimate the covariance
        source_samples = errors.iter_samples(
            np.arange(values.size), nominal=values.ravel(), kind="value", n_samples=8, random_state=6
        )
        source_draws = np.array(list(source_samples))
        expected_samples = np.array([statistic(draw) for draw in source_draws])
        covariance = np.cov(source_draws, rowvar=False)
        weights = np.full(values.size, 1 / values.size) if isinstance(operator, Interpolator) else np.ones(values.size)
        expected_variance = weights @ covariance @ weights
        expected = {
            "estimate": statistic(values),
            "mean": np.mean(expected_samples),
            "std": np.sqrt(expected_variance),
            "variance": expected_variance,
            "n_valid": 8,
        }

        # We compare the expected propagation values
        assert summary.method == "numerical"
        assert summary.error_structure is errors
        for quantity, expected_value in expected.items():
            result = getattr(summary, quantity)
            if isinstance(result, gu.Raster):
                result = result.to_nanarray()
            assert np.allclose(result, expected_value, equal_nan=True)

        # And the returned samples
        if return_samples:
            assert summary.samples is not None
            assert summary.samples.shape == (8, 1)
            assert np.allclose(summary.samples.iloc[:, 0], expected_samples, equal_nan=True)
        else:
            assert summary.samples is None


class TestPropagationSummary:
    """Test module the PropagationSummary lightweight class to analyze propagation outputs."""

    def test_propagsummary__interval(self) -> None:
        """Checks the confidence interval method of PropagationSummary."""

        # Define independent errors, variance of mean = (4 + 4) / 4
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        summary = gu.uncertainty.propagate(Mean(), data, structure)

        # Check a normal confidence interval
        interval = summary.interval(0.95)
        marginal = summary.marginal("value")
        assert interval.loc["value", "lower"] == pytest.approx(2 - 1.96 * np.sqrt(2), abs=0.001)
        assert interval.loc["value", "upper"] == pytest.approx(2 + 1.96 * np.sqrt(2), abs=0.001)
        assert marginal == {"family": "normal", "mean": 2, "std": np.sqrt(2), "status": "exact"}

    def test_propagsummary__marginal(self) -> None:
        """Checks the marginal method from PropagateSummary."""

        # We define seeded draws as reference for marginal and median
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        summary = gu.uncertainty.propagate(
            Mean(), data, structure, method="numerical", n_samples=4, random_state=6, return_samples=True
        )

        # Compare with saved draws
        assert summary.samples is not None
        pd.testing.assert_series_equal(summary.marginal("value"), summary.samples["value"])
        assert summary.quantile(0.5).loc["value"] == pytest.approx(np.median(summary.samples["value"]))

    def test_propagsummary__copies_inputs(self) -> None:
        """Checks that editing a propagation summary inputs does not propagate back to it."""

        # We define inputs
        labels = pd.Index(["left", "right"], name="output")
        selection = pd.DataFrame({"flat_index": [0, 1], "estimate": [2.0, 4.0]}, index=labels)
        samples = pd.DataFrame([[1.0, 2.0], [3.0, 6.0]], columns=labels)
        quantiles = pd.DataFrame([[2.0, 4.0]], index=[0.5], columns=labels)
        error_structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Store them in a summary
        summary = gu.PropagationSummary(
            estimate=np.array([2.0, 4.0]),
            mean=np.array([2.5, 4.5]),
            std=np.array([1.0, 2.0]),
            error_structure=error_structure,
            method="numerical",
            selection=selection,
            samples=samples,
            quantiles=quantiles,
        )

        # We change the original tables and check that it does not back-propagate
        selection.loc["left", "estimate"] = 99
        samples.loc[0, "left"] = 99
        quantiles.loc[0.5, "left"] = 99
        assert summary.selection.loc["left", "estimate"] == 2
        assert summary.samples is not None and summary.samples.loc[0, "left"] == 1
        np.testing.assert_array_equal(summary.quantile(0.5), [2, 4])

    def test_summary__circular_variance(self) -> None:
        """Checks that a circular bias wraps across a period and ordinary variance is unavailable."""

        # We define an estimate near 360 degrees and a mean near zero differ by two degrees
        selection = pd.DataFrame({"flat_index": [0], "estimate": [359.0]}, index=["bearing"])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        summary = gu.PropagationSummary(
            estimate=np.array([359.0]),
            mean=np.array([1.0]),
            std=np.array([3.0]),
            error_structure=source_error,
            method="numerical",
            selection=selection,
            output={"circular_period": 360},
        )

        # Check error on classic variance
        np.testing.assert_array_equal(summary.bias, [2])
        with pytest.raises(TypeError, match="Circular output has no ordinary variance"):
            _ = summary.variance

    def test_quantile__unwraps_circular(self) -> None:
        """Checks that circular quantiles work as intended."""

        # Define a propagation summary from a circular variable
        labels = pd.Index(["angle"])
        selection = pd.DataFrame({"flat_index": [0], "estimate": [359.0]}, index=labels)
        summary = gu.PropagationSummary(
            estimate=np.array([359.0]),
            mean=np.array([359.0]),
            std=np.array([2.0]),
            error_structure=gu.ErrorStructure([gu.ErrorComponent("measurement", 1)]),
            method="numerical",
            selection=selection,
            samples=pd.DataFrame([[357.0], [1.0]], columns=labels),
            output={"circular_period": 360},
        )

        # We unwrap 1 degree to 361 before taking the midpoint and 50% interval
        assert summary.quantile(0.5).loc["angle"] == 359
        interval = summary.interval(0.5)
        assert interval.loc["angle", "lower"] == 358
        assert interval.loc["angle", "upper"] == 360
        assert interval.attrs["coordinate_convention"] == "unwrapped_about_estimate"

    def test_marginal_quantile_interval__error_samples(self) -> None:
        """Checks that a numerical result without draws raises an error."""

        # Numerical propagation without saved draws
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        summary = gu.uncertainty.propagate(Mean(), data, structure, method="numerical", n_samples=4, random_state=6)
        with pytest.raises(ValueError, match="no saved draws or known marginal"):
            summary.marginal("value")
        with pytest.raises(ValueError, match="Quantiles require saved draws"):
            summary.quantile(0.5)
        with pytest.raises(ValueError, match="Quantiles require saved draws"):
            summary.interval()

    @pytest.mark.parametrize(
        "selection",
        [
            pd.DataFrame({"flat_index": [0, 1], "estimate": [2.0, 4.0]}, index=["left", "left"]),
            pd.DataFrame({"flat_index": [0, 1]}, index=["left", "right"]),
        ],
    )
    def test_summary__error_invalid_selection(self, selection: pd.DataFrame) -> None:
        """Checks an error is raised for duplicate output labels or missing estimates."""

        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="selection must"):
            gu.PropagationSummary(
                estimate=np.array([2.0, 4.0]),
                mean=np.array([2.0, 4.0]),
                std=np.array([1.0, 1.0]),
                error_structure=source_error,
                method="analytical",
                selection=selection,
            )

    @pytest.mark.parametrize("field", ["samples", "quantiles"])
    def test_summary__error_mismatched_labels(self, field: str) -> None:
        """Checks an error is raised when samples or quantiles have mismatched labels."""

        # Reverse the labels, which changes the meaning of each draw
        labels = pd.Index(["left", "right"], name="output")
        reverse = labels[::-1]
        selection = pd.DataFrame({"flat_index": [0, 1], "estimate": [2.0, 4.0]}, index=labels)
        tables = {
            "samples": pd.DataFrame([[2.0, 4.0]], columns=reverse),
            "quantiles": pd.DataFrame([[2.0, 4.0]], index=[0.5], columns=reverse),
        }
        options: dict[str, Any] = {field: tables[field]}

        # Check the reversed labels for both tables
        with pytest.raises(ValueError, match=field):
            gu.PropagationSummary(
                estimate=np.array([2.0, 4.0]),
                mean=np.array([2.0, 4.0]),
                std=np.array([1.0, 1.0]),
                error_structure=gu.ErrorStructure([gu.ErrorComponent("measurement", 1)]),
                method="numerical",
                selection=selection,
                **options,
            )

    @pytest.mark.parametrize(
        "method, invalid, error_type, message",
        [
            ("quantile", True, TypeError, "probability must be a scalar number"),
            ("quantile", "half", TypeError, "probability must be a scalar number"),
            ("quantile", -0.1, ValueError, "probability must be finite"),
            ("quantile", 1.1, ValueError, "probability must be finite"),
            ("quantile", np.nan, ValueError, "probability must be finite"),
            ("interval", True, TypeError, "coverage must be a scalar number"),
            ("interval", "half", TypeError, "coverage must be a scalar number"),
            ("interval", 0, ValueError, "coverage must be finite"),
            ("interval", 1, ValueError, "coverage must be finite"),
            ("interval", np.nan, ValueError, "coverage must be finite"),
        ],
    )
    def test_summary__error_invalid_probability_and_coverage(
        self, method: str, invalid: object, error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for an invalid quantile probability or interval coverage."""

        data = _local([1.0, 3.0], ["a", "b"], [[0.0], [1.0]])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        summary = gu.uncertainty.propagate(Mean(), data, source_error)

        with pytest.raises(error_type, match=message):
            getattr(summary, method)(invalid)


class TestPropagationErrors:
    """Test module for warnings/errors in propagate()."""

    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"error_structure": object()}, TypeError, "error_structure must"),
            ({"method": "invalid"}, ValueError, "method must"),
            ({"max_sample_bytes": 0}, ValueError, "max_sample_bytes must"),
            ({"operation_kwargs": {}}, TypeError, "requires a bound spatial method"),
            ({"operator": object()}, TypeError, "operator must"),
            ({"operator": np.mean}, TypeError, "operator must"),
            ({"data": []}, ValueError, "at least one LocalData target"),
        ],
    )
    def test_propagate__error_invalid_options(
        self, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid model, method, storage limits, or operator inputs."""

        data = _local([1.0, 3.0], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        defaults: dict[str, Any] = {"operator": Mean(), "data": data, "error_structure": structure}
        with pytest.raises(error_type, match=message):
            gu.uncertainty.propagate(**(defaults | options))

    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"output_labels": ["left", "left"]}, ValueError, "output_labels must contain one unique label"),
            ({"at": "left"}, TypeError, "at must be 'all' or a sequence"),
            ({"at": ["right", "right"]}, ValueError, "at must name distinct existing"),
            ({"at": ["missing"]}, ValueError, "at must name distinct existing"),
        ],
    )
    def test_propagate__error_invalid_output(
        self, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for duplicate labels or unknown selected outputs."""

        first = _local([1.0, 2.0], ["a", "b"], [[0, 0], [1, 0]])
        second = _local([2.0, 3.0], ["b", "c"], [[1, 0], [2, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        defaults: dict[str, Any] = {"output_labels": ["left", "right"]}

        with pytest.raises(error_type, match=message):
            gu.uncertainty.propagate(Mean(), [first, second], structure, **(defaults | options))


class TestSimulationErrors:
    """Test module for warnings/errors in simulate()."""

    @pytest.mark.parametrize(
        "bad_draw, message", [(np.array([1.0, 2.0]), "changed the output shape"), (np.nan, "no finite values")]
    )
    def test_simulate__error_bad_calculation_result(self, bad_draw: object, message: str) -> None:
        """Checks an error is raised for a draw with the wrong shape or no finite values."""

        # A scalar output cannot accept an array or a missing value
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match=message):
            simulate(
                estimate=np.array(0.0),
                draw_error=lambda _rng: bad_draw,
                calculate=lambda draw: draw,
                error_structure=structure,
                selection=selection,
                n_samples=2,
            )

    def test_simulate__error_insufficient_successful_draws(self) -> None:
        """Checks an error is raised when fewer than two draws succeed."""

        # We define inputs so that the second calculation raises en error after the first returns a finite value
        draws = iter((1.0, np.nan))
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.warns(UserWarning, match="Simulation 2 of 2 failed"):
            with pytest.raises(RuntimeError, match="Only 1 of 2 simulations succeeded"):
                simulate(
                    estimate=np.array(0.0),
                    draw_error=lambda _rng: next(draws),
                    calculate=lambda draw: draw,
                    error_structure=structure,
                    selection=selection,
                    n_samples=2,
                    on_error="warn",
                )

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"n_samples": 1}, "n_samples must be an integer"),
            ({"n_samples": True}, "n_samples must be an integer"),
            ({"on_error": "ignore"}, "on_error must be"),
            ({"max_sample_bytes": 0}, "max_sample_bytes must be positive"),
            ({"quantiles": (1.1,)}, "quantiles must be finite probabilities"),
            ({"quantiles": (0.5, 0.5)}, "quantiles must be distinct"),
            ({"output": {"circular_period": 0}}, "finite positive period"),
            ({"return_samples": True, "max_sample_bytes": 1}, "Selected sample buffer needs"),
        ],
    )
    def test_simulate__error_invalid_options(self, options: dict[str, object], message: str) -> None:
        """Checks an error is raised for invalid draw counts, probabilities, or storage limits."""

        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        arguments: dict[str, object] = {"n_samples": 2, "selection": selection}
        arguments.update(options)
        with pytest.raises(ValueError, match=message):
            simulate(
                estimate=np.array(0.0),
                draw_error=lambda _rng: pytest.fail("drawn before validation"),
                calculate=lambda draw: draw,
                error_structure=structure,
                **arguments,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "selection, message",
        [
            (pd.DataFrame({"flat_index": [0]}), "selection must contain"),
            (
                pd.DataFrame({"flat_index": [0, 0], "estimate": [0.0, 0.0]}, index=["same", "same"]),
                "selection keys must be unique",
            ),
            (pd.DataFrame({"flat_index": [1], "estimate": [0.0]}), "position outside the output"),
        ],
    )
    def test_simulate__error_invalid_selection(self, selection: pd.DataFrame, message: str) -> None:
        """Checks an error is raised for duplicate output labels or positions outside the output."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match=message):
            simulate(
                estimate=np.array(0.0),
                draw_error=lambda _rng: pytest.fail("drawn before validation"),
                calculate=lambda draw: draw,
                error_structure=structure,
                selection=selection,
                n_samples=2,
            )
