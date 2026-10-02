"""Tests for analytical and numerical uncertainty propagation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from affine import Affine

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.operators import Interpolator, LinearCoefficients, LocalData, Reducer
from geoutils.operators.neighbours import GridNeighbours
from geoutils.operators.reducer import Mean
from geoutils.stats.variography import VariogramModel
from geoutils.uncertainty.propagation import simulate
from tests.operator_helpers import LocalMeanInterpolator


class NonlinearMeanSquare(Reducer):
    """Square the mean so we can test uncertainty propagation through a nonlinear function."""

    def reduce(self, data: LocalData) -> float:
        """Return the square of the mean of the source values."""

        return float(np.mean(data.values) ** 2)


class CountingMean(Reducer):
    """Count reductions so we can check whether a supplied original result is recalculated."""

    def __init__(self) -> None:
        """Start with no completed reductions."""

        self.calls = 0

    def reduce(self, data: LocalData) -> float:
        """Count this call and return the mean of the source values."""

        self.calls += 1
        return float(np.mean(data.values))


class FirstValue(Interpolator):
    """Select the first input with a zero coefficient on all remaining values."""

    default_nodata_propagation = "propagate"

    def coefficients(self, data: LocalData) -> LinearCoefficients:
        """Give the first source a weight of one and every other source a weight of zero."""

        weights = np.zeros(len(data.values), dtype=float)
        weights[0] = 1
        return LinearCoefficients(weights)


def _local(values: list[float], source_ids: list[str], coordinates: list[list[float]]) -> LocalData:
    """Create finite local data for small propagation calculations."""

    return LocalData(
        values=np.asarray(values, dtype=float),
        valid=np.ones(len(values), dtype=bool),
        source_ids=np.asarray(source_ids),
        coordinates=np.asarray(coordinates, dtype=float),
    )


class TestAnalyticalOperatorPropagation:
    """Test module for linear weights, repeated observations, and nodata."""

    def test_propagate__exact_coefficients_preserve_source_identity(self) -> None:
        """Checks that repeated IDs are fully dependent while coincident distinct IDs remain independent."""

        # Coincident observations with distinct IDs, then repeated use of ID a
        first = _local([10, 20], ["a", "b"], [[0, 0], [0, 0]])
        second = _local([10, 10], ["a", "a"], [[0, 0], [0, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Distinct IDs have separate variance; repeated a contributes one full source error
        summary = gu.uncertainty.propagate(Mean(), [first, second], structure)
        np.testing.assert_allclose(summary.estimate, [15, 10])
        np.testing.assert_allclose(summary.variance, [2, 4])

    def test_propagate__aligns_grouped_magnitudes_by_source_id(self) -> None:
        """Checks that named predictors assign each source its own grouped error magnitude before averaging."""

        # Error magnitudes grouped by slope, source values in B/A order
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

    def test_propagate__array_predictors_follow_numeric_source_order(self) -> None:
        """Checks that array predictors follow numeric source IDs (1, 2, 10), not text order."""

        # Grouped errors in ascending ID order, source values in reverse order
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

    def test_propagate__zero_coefficient_ignores_nodata(self) -> None:
        """Checks that invalid input with exactly zero coefficient does not invalidate the result."""

        # Missing second source with zero weight in FirstValue()
        data = LocalData(
            values=np.array([3.0, np.nan]),
            valid=np.array([True, False]),
            source_ids=np.array(["used", "unused"]),
            coordinates=np.array([[0.0, 0.0], [1.0, 0.0]]),
        )
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])

        # Check value and error from first source only
        summary = gu.uncertainty.propagate(FirstValue(), data, structure)
        assert summary.estimate == pytest.approx(3)
        assert summary.std == pytest.approx(0.5)

    def test_propagate__missing_source_has_undefined_uncertainty(self) -> None:
        """Checks that a target with no valid source has missing original and propagated results."""

        # Single missing source
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
        """Checks that selected outputs determine the analytical quantile columns."""

        # Two means use b, but their own variances depend on two independent sources each
        first = _local([1, 2], ["a", "b"], [[0, 0], [1, 0]])
        second = _local([2, 3], ["b", "c"], [[1, 0], [2, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Select both outputs or only the second while calculating the same two means
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

    def test_propagate__error_analytical_samples(self) -> None:
        """Checks an error is raised when analytical propagation requests random samples."""

        # Mean with exact coefficients for analytical propagation
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check error for samples requested with analytical propagation
        with pytest.raises(ValueError, match="return_samples cannot be used with analytical"):
            gu.uncertainty.propagate(Mean(), data, structure, method="analytical", return_samples=True)

    def test_propagate__error_invalid_nominal_estimate(self) -> None:
        """Checks an error is raised for more supplied results than targets."""

        # Two observations in one group (one output)
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check error for two supplied results instead of one
        with pytest.raises(ValueError, match="one value per LocalData target"):
            gu.uncertainty.propagate(Mean(), data, structure, nominal_estimate=[2, 3])


class TestNumericalOperatorPropagation:
    """Test module for nonlinear methods and repeatable errors when targets are reordered."""

    def test_propagate__nonlinear_operator_uses_numerical_fallback(self) -> None:
        """Checks that propagate() simulates the mean and spread of a nonlinear result."""

        # Small independent errors for nonlinear mean-square calculation
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

    def test_propagate__auto_samples_use_numerical_draws(self) -> None:
        """Checks that automatic propagation saves samples by choosing numerical calculation for a mean."""

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

    def test_propagate__supplied_nominal_is_not_recalculated(self) -> None:
        """Checks that a supplied original result avoids a duplicate reduction before numerical draws."""

        # Count operator calls with precomputed mean
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        operator = CountingMean()

        # Check supplied result and one reduce() call per draw
        summary = gu.uncertainty.propagate(
            operator, data, structure, method="numerical", nominal_estimate=2, n_samples=3, random_state=6
        )
        assert summary.estimate == 2
        assert operator.calls == 3

    def test_propagate__masked_source_stays_invalid_in_numerical_draws(self) -> None:
        """Checks that a masked finite value is excluded from source errors and every simulated reduction."""

        # Masked finite value (only first source enters draws)
        values = np.ma.array([2.0, 100.0], mask=[False, True])
        data = LocalData(
            values=values,
            valid=np.ones(2, dtype=bool),
            source_ids=np.asarray(["used", "masked"]),
        )
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Check identical draws with zero error (square of unmasked value)
        summary = gu.uncertainty.propagate(
            NonlinearMeanSquare(), data, errors, method="numerical", n_samples=4, return_samples=True
        )
        assert summary.estimate == 4
        assert summary.samples is not None
        np.testing.assert_array_equal(summary.samples.to_numpy(), np.full((4, 1), 4.0))

    def test_propagate__target_order_does_not_change_source_realizations(self) -> None:
        """Checks that reversing the target order does not change the errors generated with the same seed."""

        # Overlapping targets evaluated in forward/reverse order
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

        # Check matching draws per target after reordering
        assert forward.samples is not None and reversed_order.samples is not None
        np.testing.assert_array_equal(forward.samples["first"], reversed_order.samples["first"])
        np.testing.assert_array_equal(forward.samples["second"], reversed_order.samples["second"])

    def test_propagate__error_samples_without_valid_sources(self) -> None:
        """Checks an error is raised when sampling from a target with no valid source."""

        # Missing source with no valid observation to sample
        data = LocalData(values=np.array([np.nan]), valid=np.array([False]), source_ids=np.array(["missing"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check sample request error for missing source
        with pytest.raises(ValueError, match="at least one valid source"):
            gu.uncertainty.propagate(Mean(), data, structure, return_samples=True)


class TestSimulation:
    """Test module for partial draws, circular results, and skipped calculations in simulate()."""

    def test_simulate__partial_draws_use_each_outputs_finite_values(self) -> None:
        """Checks that each output uses its own finite draws for its mean and spread."""

        # Three draws, with the second output missing from the middle draw
        draws = iter((np.array([1.0, 2.0]), np.array([3.0, np.nan]), np.array([5.0, 6.0])))
        selection = pd.DataFrame({"flat_index": [0, 1], "estimate": [0.0, 0.0]}, index=["first", "second"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Save draws and quantiles for both outputs
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

        # The first output has three finite draws and the second has two
        np.testing.assert_array_equal(summary.n_valid, [3, 2])
        np.testing.assert_allclose(summary.mean, [3, 4])
        np.testing.assert_allclose(summary.std, [2, np.sqrt(8)])
        assert summary.samples is not None
        expected_samples = np.array([[1, 2], [3, np.nan], [5, 6]])
        np.testing.assert_allclose(summary.samples.to_numpy(), expected_samples, equal_nan=True)
        assert summary.quantiles is not None
        np.testing.assert_allclose(summary.quantiles.loc[0.5], [3, 4])

    def test_simulate__circular_draws_cross_zero(self) -> None:
        """Checks that angles on either side of zero have a near-zero mean and a narrow spread."""

        # Angles 359 and 1 degrees are two degrees apart across the wrap point
        draws = iter((359.0, 1.0))
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["angle"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Request a quantile in the same unwrapped frame as the original angle
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

        # Circular mean is zero; unwrapped selected values are -1 and 1 degrees
        assert summary.mean % 360 == pytest.approx(0, abs=1e-12)
        expected_resultant = np.cos(np.deg2rad(1))
        assert summary.resultant_length == pytest.approx(expected_resultant)
        assert summary.std == pytest.approx(180 / np.pi * np.sqrt(-2 * np.log(expected_resultant)))
        assert summary.quantiles is not None
        assert summary.quantiles.loc[0.5, "angle"] == pytest.approx(0)
        assert summary.samples is not None
        np.testing.assert_array_equal(summary.samples["angle"], [359, 1])

    def test_simulate__warns_and_skips_failed_draw(self) -> None:
        """Checks that a failed calculation is recorded without changing the statistics of successful draws."""

        # A negative draw makes the calculation fail between two finite results
        draws = iter((1.0, -1.0, 3.0))
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        def calculate(draw: float) -> float:
            """Reject the negative draw and return the others unchanged."""

            if draw < 0:
                raise ValueError("negative draw")
            return draw

        # Successful draws 1 and 3 have mean 2 and sample standard deviation sqrt(2)
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
        assert summary.n_success == 2
        assert summary.failures == {2: "negative draw"}
        assert summary.mean == pytest.approx(2)
        assert summary.std == pytest.approx(np.sqrt(2))
        assert summary.samples is not None
        np.testing.assert_allclose(summary.samples["value"], [1, np.nan, 3], equal_nan=True)

    def test_simulate__selected_output_has_no_finite_draws(self) -> None:
        """Checks that an always missing output has an undefined mean and zero valid draws."""

        # Each draw has one finite output, but the selected second output is always missing
        draws = iter((np.array([1.0, np.nan]), np.array([3.0, np.nan])))
        selection = pd.DataFrame({"flat_index": [1], "estimate": [np.nan]}, index=["missing"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check the finite first output and always missing second output
        summary = simulate(
            estimate=np.array([0.0, np.nan]),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: draw,
            error_structure=structure,
            selection=selection,
            n_samples=2,
        )
        np.testing.assert_allclose(summary.mean, [2, np.nan], equal_nan=True)
        np.testing.assert_array_equal(summary.n_valid, [2, 0])


class TestCallablePropagation:
    """Test module for complete callable calculations and selected spatial output distributions."""

    def test_propagate__same_source_value_cancels_in_complete_calculation(self) -> None:
        """Checks that repeated uses of one uncertain input share the same draw."""

        # Independent named inputs for repeated-use calculation
        labels = pd.Index(["a", "b"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        source = pd.Series([10.0, 20.0], index=labels)

        # Calculate difference/sum from same source draw
        def calculate(values: pd.Series) -> pd.Series:
            """Return a cancelling difference and the sum of both inputs."""

            return pd.Series({"difference": values["a"] - values["a"], "sum": values["a"] + values["b"]})

        summary = gu.uncertainty.propagate(
            calculate,
            source,
            structure,
            n_samples=100,
            random_state=9,
            return_samples=True,
        )

        # Check exact cancellation in difference, positive spread in sum
        assert summary.error_structure is structure
        assert summary.samples is not None
        np.testing.assert_array_equal(summary.samples["difference"], 0)
        assert summary.std["sum"] > 0

    def test_propagate__large_raster_selects_only_requested_cell(self) -> None:
        """Checks that a large raster result saves draws only for explicitly selected cells."""

        # Zero error for exact selected value (row 3, column 4)
        values = np.arange(400, dtype=float).reshape(20, 20)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 20), 32632)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Select one output cell to limit saved draws
        summary = gu.uncertainty.propagate(
            lambda source: source,
            raster,
            structure,
            at=[(3, 4)],
            n_samples=2,
            return_samples=True,
            random_state=4,
        )

        # Check full raster mean and samples for selected cell only
        assert isinstance(summary.mean, gu.Raster)
        assert summary.samples is not None
        assert summary.samples.shape == (2, 1)
        np.testing.assert_array_equal(summary.samples[(3, 4)], [64, 64])

    def test_propagate__dataframe_output_selects_row_and_column(self) -> None:
        """Checks that selected DataFrame results use their row and column labels in saved draws."""

        # Zero error makes every draw equal the original two-column calculation
        source = np.array([2.0, 5.0])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        def calculate(values: NDArrayNum) -> pd.DataFrame:
            """Put the source values in a labelled row."""

            return pd.DataFrame([[values[0], values[1]]], index=["row"], columns=["left", "right"])

        # Select the right column by both labels, then compare its saved realizations
        summary = gu.uncertainty.propagate(
            calculate,
            source,
            structure,
            at=[("row", "right")],
            n_samples=3,
            return_samples=True,
        )
        assert summary.samples is not None
        np.testing.assert_array_equal(summary.samples[("row", "right")], [5, 5, 5])
        assert summary.selection["flat_index"].iloc[0] == 1

    @pytest.mark.parametrize(
        "result, at",
        [
            (np.array([1.0, 2.0]), "missing"),
            (np.array([1.0, 2.0]), [0, 0]),
            (np.array([1.0, 2.0]), [2]),
            (np.array([[1.0, 2.0]]), [0]),
            (pd.DataFrame([[1.0, 2.0]], index=["row"], columns=["left", "right"]), [("row", "missing")]),
        ],
    )
    def test_propagate__error_invalid_callable_output_selection(self, result: object, at: object) -> None:
        """Checks an error is raised for duplicate or unknown callable output labels."""

        # A fixed calculation isolates validation of its requested output labels
        source = np.array([0.0])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Reject unknown, repeated, or wrongly shaped labels before simulation
        with pytest.raises(ValueError, match="at must"):
            gu.uncertainty.propagate(
                lambda _values: result,
                source,
                structure,
                at=at,  # type: ignore[arg-type]
                n_samples=2,
                return_samples=True,
            )

    def test_propagate__error_callable_requires_numerical_method(self) -> None:
        """Checks an error is raised for analytical propagation through an arbitrary callable."""

        # A deterministic sum still has no coefficients for an arbitrary callable
        source = np.array([1.0, 2.0])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Reject analytical propagation before drawing errors
        with pytest.raises(NotImplementedError, match="needs explicit derivatives"):
            gu.uncertainty.propagate(lambda values: values.sum(), source, structure, method="analytical")

    def test_propagate__error_selected_samples_exceed_memory_limit(self) -> None:
        """Checks an error is raised before drawing samples that exceed the memory limit."""

        # Two float64 draws need 16 bytes (budget: 8)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        source = np.array([10.0])

        # Check error for sample table exceeding memory limit
        with pytest.raises(ValueError, match="max_sample_bytes"):
            gu.uncertainty.propagate(
                lambda values: values,
                source,
                structure,
                n_samples=2,
                return_samples=True,
                max_sample_bytes=8,
            )

    @pytest.mark.parametrize(
        "option",
        [{"nodata_propagation": "ignore"}, {"nominal_estimate": 3}, {"output_labels": ["value"]}],
    )
    def test_propagate__error_local_options_with_callable(self, option: dict[str, Any]) -> None:
        """Checks an error is raised for local operator options on a callable calculation."""

        # Whole-array callable with each local-operator option
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        source = np.array([1.0, 2.0])

        # Check unsupported options before drawing errors
        with pytest.raises(ValueError, match="apply only to local operators"):
            gu.uncertainty.propagate(
                lambda values: values.sum(),
                source,
                structure,
                **option,
            )


class TestSpatialPropagation:
    """Test module for weighted estimates, geometric coefficients, and ordinary spatial return types."""

    @pytest.mark.parametrize("engine", ["scipy", "numba"])
    @pytest.mark.parametrize("method", ["nearest", "idw"])
    def test_grid__rectangular_pixel_distances(self, engine: str, method: str) -> None:
        """Checks that adding homogeneous errors preserves each method's distance units on rectangular pixels."""

        # The first point is nearest in coordinate units, while the second is nearest in output-pixel units
        if engine == "numba":
            pytest.importorskip("numba")
        points = gu.PointCloud.from_xyz([0.0, 4.0], [1.0, 0.0], [10.0, 20.0], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 0, 10, 1), crs=32631)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        kwargs = {"ref": reference, "resampling": method, "dist_nodata_pixel": 2, "engine": engine}

        expected = points.grid(**kwargs)
        result = points.grid(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(points.grid, error_structure=errors, operation_kwargs=kwargs)

        # Equal observation errors preserve the distance coefficients, and propagation uses the same result
        np.testing.assert_allclose(result.to_nanarray(), expected.to_nanarray(), rtol=1e-14)
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), result.to_nanarray())
        np.testing.assert_allclose(summary.mean.to_nanarray(), result.to_nanarray(), rtol=1e-14)

    def test_resample_at_points__correlated_fit(self) -> None:
        """Checks that a raster window uses correlated errors for both its mean and propagated variance."""

        pytest.importorskip("skgstat")

        # Two-cell neighborhood with correlated errors (equal geometric weights)
        raster = gu.Raster.from_array(np.array([[10.0, 20.0]]), rio.transform.from_origin(0, 1, 1, 1), crs=32631)
        correlation = VariogramModel("spherical", effective_range=5, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])
        operator = Mean(GridNeighbours(offsets=((0, 0), (0, 1))))
        kwargs = {"points": ([0.5], [0.5]), "method": operator, "as_array": True}

        # Compare direct/propagated results with generalized least squares
        result = raster.resample_at_points(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(raster.resample_at_points, error_structure=errors, operation_kwargs=kwargs)
        covariance = errors.to_covariance_matrix(np.array([[0.5, 0.5], [1.5, 0.5]]))
        weights = np.linalg.solve(covariance, np.ones(2))
        weights /= weights.sum()

        assert result == pytest.approx(weights @ [10, 20])
        assert summary.estimate == result
        assert summary.variance == pytest.approx(weights @ covariance @ weights)
        assert summary.variance > 2

    def test_reproject__integer_input_has_fractional_uncertainty(self) -> None:
        """Checks that a fractional error magnitude remains floating point when the source raster stores integers."""

        # Averaging four independent errors of magnitude one gives a standard deviation of one half
        raster = gu.Raster.from_array(
            np.array([[2, 4], [6, 8]], dtype=np.int16), rio.transform.from_origin(0, 2, 1, 1), crs=32631, nodata=-9999
        )
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 2, 2, 2), crs=32631)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        kwargs = {"ref": reference, "resampling": Mean()}

        summary = gu.uncertainty.propagate(raster.reproject, error_structure=errors, operation_kwargs=kwargs)

        assert summary.estimate.to_nanarray()[0, 0] == 5
        assert np.issubdtype(summary.std.data.dtype, np.floating)
        assert summary.std.to_nanarray()[0, 0] == 0.5

    def test_reproject__bilinear_keeps_geometric_coefficients(self) -> None:
        """Checks that bilinear interpolation uses geometric weights with spatially correlated errors."""

        pytest.importorskip("skgstat")

        # Target halfway between four cells (bilinear weights 1/4 each)
        raster = gu.Raster.from_array(
            np.array([[0.0, 4.0], [8.0, 12.0]]), rio.transform.from_origin(0, 2, 1, 1), crs=32631
        )
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.5, 1.5, 1, 1), crs=32631)
        correlation = VariogramModel("spherical", effective_range=5, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])
        kwargs = {"ref": reference, "resampling": "bilinear"}

        result = raster.reproject(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(raster.reproject, error_structure=errors, operation_kwargs=kwargs)
        centers = np.array([[0.5, 1.5], [1.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
        covariance = errors.to_covariance_matrix(centers)
        weights = np.full(4, 0.25)

        # Check bilinear value and variance from geometric weights
        assert result.to_nanarray()[0, 0] == 6
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), result.to_nanarray())
        assert summary.variance.to_nanarray()[0, 0] == pytest.approx(weights @ covariance @ weights)

    def test_reproject__custom_interpolator_multiband_source_ids(self) -> None:
        """Checks that uncertainty uses each source band's cell IDs once for a custom interpolator."""

        # Give the two bands nine distinct observations with independent measurement errors
        first_band = np.arange(9, dtype=float).reshape(3, 3)
        source = gu.Raster.from_array(
            np.stack((first_band, first_band + 9)),
            rio.transform.from_origin(0, 3, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            rio.transform.from_origin(1, 2, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Both bands use the same three by three window, with distinct IDs 0-8 and 9-17
        nominal = source.reproject(resampling=LocalMeanInterpolator(), ref=reference, error_structure=errors)
        summary = gu.uncertainty.propagate(
            source.reproject,
            error_structure=errors,
            operation_kwargs={"resampling": LocalMeanInterpolator(), "ref": reference},
            n_samples=4,
        )
        np.testing.assert_allclose(nominal.to_nanarray().reshape(-1), [4, 13])
        np.testing.assert_allclose(summary.estimate.to_nanarray().reshape(-1), [4, 13])
        np.testing.assert_array_equal(summary.n_valid.to_nanarray().reshape(-1), [4, 4])


# Tests for intervals and marginals in propagation results.


class TestPropagationSummary:
    """Test module for analytical intervals and numerical summaries with or without stored draws."""

    def test_interval__analytical_mean_has_normal_bounds(self) -> None:
        """Checks that an analytical mean has symmetric normal bounds and an exact normal marginal."""

        # Independent errors: variance of mean = (4 + 4) / 4
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        summary = gu.uncertainty.propagate(Mean(), data, structure)

        # Check normal interval using rounded 1.96 multiplier
        interval = summary.interval(0.95)
        marginal = summary.marginal("value")
        assert interval.loc["value", "lower"] == pytest.approx(2 - 1.96 * np.sqrt(2), abs=0.001)
        assert interval.loc["value", "upper"] == pytest.approx(2 + 1.96 * np.sqrt(2), abs=0.001)
        assert marginal == {"family": "normal", "mean": 2, "std": np.sqrt(2), "status": "exact"}

    def test_marginal__numerical_samples_match_saved_draws(self) -> None:
        """Checks that a saved numerical marginal and its median come from the same output draws."""

        # Save seeded draws as reference for marginal and median
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        summary = gu.uncertainty.propagate(
            Mean(), data, structure, method="numerical", n_samples=4, random_state=6, return_samples=True
        )

        # Compare marginal and median with saved draws
        assert summary.samples is not None
        pd.testing.assert_series_equal(summary.marginal("value"), summary.samples["value"])
        assert summary.quantile(0.5).loc["value"] == pytest.approx(np.median(summary.samples["value"]))

    def test_propagate__numerical_result_keeps_initial_model(self) -> None:
        """Checks that numerical output moments stay in the result while the source model stays unchanged."""

        # Zero-mean, zero-variance errors leave source values unchanged
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])
        data = LocalData(values=np.array([10.0, 20.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))

        # Check output moments and original source model
        summary = gu.uncertainty.propagate(Mean(), data, structure, method="numerical", n_samples=4)
        assert summary.error_structure is structure
        assert summary.estimate == pytest.approx(15)
        assert summary.mean == pytest.approx(15)
        assert summary.std == pytest.approx(0)

    def test_summary__copies_labelled_outputs(self) -> None:
        """Checks that editing input tables does not change a stored propagation summary."""

        # Label two outputs and their saved samples and quantiles
        labels = pd.Index(["left", "right"], name="output")
        selection = pd.DataFrame({"flat_index": [0, 1], "estimate": [2.0, 4.0]}, index=labels)
        samples = pd.DataFrame([[1.0, 2.0], [3.0, 6.0]], columns=labels)
        quantiles = pd.DataFrame([[2.0, 4.0]], index=[0.5], columns=labels)
        error_structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Store the tables in a summary
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

        # Change the original tables and check the stored values
        selection.loc["left", "estimate"] = 99
        samples.loc[0, "left"] = 99
        quantiles.loc[0.5, "left"] = 99
        assert summary.selection.loc["left", "estimate"] == 2
        assert summary.samples is not None and summary.samples.loc[0, "left"] == 1
        np.testing.assert_array_equal(summary.quantile(0.5), [2, 4])

    def test_summary__circular_bias_has_no_linear_variance(self) -> None:
        """Checks that circular bias wraps across a period and ordinary variance is unavailable."""

        # An estimate near 360 degrees and a mean near zero differ by two degrees
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

        # Wrap the mean difference instead of subtracting angles as linear numbers
        np.testing.assert_array_equal(summary.bias, [2])
        with pytest.raises(TypeError, match="Circular output has no ordinary variance"):
            _ = summary.variance

    def test_quantile__unwraps_circular_draws_around_estimate(self) -> None:
        """Checks that circular quantiles place draws on the same turn as the nominal angle."""

        # Draws of 357 and 1 degrees lie two degrees either side of an estimate at 359
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

        # Unwrap 1 degree to 361 before taking the midpoint and 50% interval
        assert summary.quantile(0.5).loc["angle"] == 359
        interval = summary.interval(0.5)
        assert interval.loc["angle", "lower"] == 358
        assert interval.loc["angle", "upper"] == 360
        assert interval.attrs["coordinate_convention"] == "unwrapped_about_estimate"

    def test_marginal__error_numerical_without_samples(self) -> None:
        """Checks that a numerical result without saved draws does not claim an exact normal marginal."""

        # Numerical propagation without saved draws
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        summary = gu.uncertainty.propagate(Mean(), data, structure, method="numerical", n_samples=4, random_state=6)

        # Check errors for distribution queries without saved draws
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
    def test_summary__error_mismatched_output_labels(self, field: str) -> None:
        """Checks an error is raised when samples or quantiles have mismatched output labels."""

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

        # Normal mean provides a valid summary for either method
        data = _local([1.0, 3.0], ["a", "b"], [[0.0], [1.0]])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        summary = gu.uncertainty.propagate(Mean(), data, source_error)

        # Check the requested method with one invalid value
        with pytest.raises(error_type, match=message):
            getattr(summary, method)(invalid)


class TestSpatialPropagationChunked:
    """Test module for lazy local uncertainty and stable source IDs across spatial and band chunks."""

    def test_reproject__weighted_band_and_spatial_chunks(self) -> None:
        """Checks that chunked weighted reprojection stays lazy and agrees exactly with eager source IDs."""

        # Different values/errors per band to check source IDs across chunks
        pytest.importorskip("dask")
        from dask.callbacks import Callback

        values = np.arange(60.0, dtype=float).reshape(2, 5, 6)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32631, nodata=-9999)
        reference = gu.Raster.from_array(np.zeros((5, 6)), raster.transform, crs=raster.crs)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [30, 30]}, index=pd.Index([0.0, 1.0], name="band"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        predictors = {"band": {source_id: float(source_id // 30) for source_id in range(values.size)}}
        kwargs = {"ref": reference, "resampling": Mean(GridNeighbours(size=3)), "nodata_propagation": "ignore"}
        expected = gu.uncertainty.propagate(
            raster.reproject, error_structure=errors, operation_kwargs=kwargs, predictors=predictors
        )

        # 1 x 2 x 4 chunks (band/Y/X), crossing neighborhoods with shorter edge chunks
        lazy = raster.to_xarray().chunk({"band": 1, "y": 2, "x": 4})
        tasks = []
        with Callback(posttask=lambda *args: tasks.append(args[0])):
            result = gu.uncertainty.propagate(
                lazy.rst.reproject, error_structure=errors, operation_kwargs=kwargs, predictors=predictors
            )
        assert not tasks
        assert hasattr(lazy.data, "compute")
        for quantity in ("estimate", "mean", "std"):
            output = getattr(result, quantity)
            assert hasattr(output.data, "compute")
            np.testing.assert_array_equal(output.compute().values, getattr(expected, quantity).to_nanarray())


class TestPropagationValidation:
    """Test module for invalid global options and output selections in propagate()."""

    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"error_structure": object()}, TypeError, "error_structure must"),
            ({"method": "invalid"}, ValueError, "method must"),
            ({"max_sample_bytes": 0}, ValueError, "max_sample_bytes must"),
            ({"operation_kwargs": {}}, TypeError, "requires a bound spatial method"),
            ({"operator": object()}, TypeError, "operator must"),
            ({"circular_period": 360}, ValueError, "Circular outputs require"),
            ({"data": []}, ValueError, "at least one LocalData target"),
        ],
    )
    def test_propagate__error_invalid_global_options(
        self, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid model, method, storage limits, or operator inputs."""

        # One finite mean provides a valid baseline for each changed option
        data = _local([1.0, 3.0], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        defaults: dict[str, Any] = {"operator": Mean(), "data": data, "error_structure": structure}

        # Each option must fail before a propagation result is produced
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
    def test_propagate__error_invalid_local_output_selection(
        self, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for duplicate labels or unknown selected outputs."""

        # Two means provide left and right output labels
        first = _local([1.0, 2.0], ["a", "b"], [[0, 0], [1, 0]])
        second = _local([2.0, 3.0], ["b", "c"], [[1, 0], [2, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        defaults: dict[str, Any] = {"output_labels": ["left", "right"]}

        # Check selected output labels before storing draws
        with pytest.raises(error_type, match=message):
            gu.uncertainty.propagate(Mean(), [first, second], structure, **(defaults | options))


class TestSimulationErrors:
    """Test module for invalid draws, options, and selections in simulate()."""

    @pytest.mark.parametrize(
        "bad_draw, message", [(np.array([1.0, 2.0]), "changed the output shape"), (np.nan, "no finite values")]
    )
    def test_simulate__error_bad_calculation_result(self, bad_draw: object, message: str) -> None:
        """Checks an error is raised for a draw with the wrong shape or no finite values."""

        # A scalar output cannot accept an array or a missing value
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Raise the validation error from the first draw
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

        # Second calculation raises after the first returns a finite value
        draws = iter((1.0, np.nan))
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # A warning records the bad draw, then the insufficient sample count raises
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

        # One selected scalar result with otherwise valid simulation options
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        arguments: dict[str, object] = {"n_samples": 2, "selection": selection}
        arguments.update(options)

        # Reject invalid options before calling the draw function
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

        # Reject malformed selections before sampling a scalar calculation
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
