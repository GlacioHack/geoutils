"""Tests for analytical and numerical propagation through local operators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from affine import Affine

import geoutils as gu
from geoutils.operators import Interpolator, LinearCoefficients, LocalData, Reducer
from geoutils.operators.reducer import Mean


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
    """Test module for linear weights, repeated observations, nodata, and covariance between results."""

    def test_propagate__exact_coefficients_preserve_source_identity(self) -> None:
        """Checks that repeated IDs are fully dependent while coincident distinct IDs remain independent."""

        # Coincident observations with distinct IDs, then repeated use of ID a
        first = _local([10, 20], ["a", "b"], [[0, 0], [0, 0]])
        second = _local([10, 10], ["a", "a"], [[0, 0], [0, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Reference covariance: independent source variances weighted by mean coefficients
        summary = gu.uncertainty.propagate(Mean(), [first, second], structure, return_covariance=True)
        coefficient_matrix = np.array([[0.5, 0.5], [1.0, 0.0]])
        expected_covariance = coefficient_matrix @ np.diag([4.0, 4.0]) @ coefficient_matrix.T
        np.testing.assert_allclose(summary.estimate, [15, 10])
        np.testing.assert_allclose(summary.std, np.sqrt(np.diag(expected_covariance)))
        assert summary.covariance is not None
        np.testing.assert_allclose(summary.covariance, expected_covariance)

    def test_propagate__uses_gaussian_error_mean_and_covariance(self) -> None:
        """Checks that an affine operator propagates finite Gaussian mean and covariance exactly."""

        # Labelled Gaussian errors with nonzero means and covariance
        labels = pd.Index(["a", "b"])
        covariance = pd.DataFrame([[4.0, 1.0], [1.0, 9.0]], index=labels, columns=labels)
        error_mean = pd.Series([1.0, -2.0], index=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance, mean=error_mean)
        data = _local([10, 20], ["a", "b"], [[0, 0], [1, 0]])

        # Check generalized least-squares weights and propagated moments
        summary = gu.uncertainty.propagate(Mean(), data, structure, return_covariance=True)
        weights = np.array([8 / 11, 3 / 11])
        assert summary.estimate == pytest.approx(weights @ data.values)
        assert summary.mean == pytest.approx(weights @ (data.values + error_mean))
        assert summary.variance == pytest.approx(weights @ covariance @ weights)
        assert summary.quantile(0.5).iloc[0] == pytest.approx(summary.mean)

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

        # Check NaN estimate, moments and covariance
        summary = gu.uncertainty.propagate(Mean(), data, structure, return_covariance=True)
        assert np.isnan(summary.estimate)
        assert np.isnan(summary.mean)
        assert np.isnan(summary.std)
        assert summary.covariance is not None and np.isnan(summary.covariance.iloc[0, 0])

    def test_propagate__error_analytical_samples(self) -> None:
        """Checks that analytical propagation rejects a request for random output samples."""

        # Mean with exact coefficients for analytical propagation
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check error for samples requested with analytical propagation
        with pytest.raises(ValueError, match="return_samples cannot be used with analytical"):
            gu.uncertainty.propagate(Mean(), data, structure, method="analytical", return_samples=True)

    def test_propagate__error_invalid_nominal_estimate(self) -> None:
        """Checks that supplied original results contain exactly one value per target."""

        # Two observations in one group (one output)
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check error for two supplied results instead of one
        with pytest.raises(ValueError, match="one value per LocalData target"):
            gu.uncertainty.propagate(Mean(), data, structure, nominal_estimate=[2, 3])

    def test_propagate__selected_covariance_keeps_initial_error_structure(self) -> None:
        """Checks that selected joint results refer to the original error model."""

        # Overlapping means with shared source error
        first = _local([1, 2], ["a", "b"], [[0, 0], [1, 0]])
        second = _local([2, 3], ["b", "c"], [[1, 0], [2, 0]])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        summary = gu.uncertainty.propagate(
            Mean(), [first, second], source_error, output_labels=["x", "y"], return_covariance=True
        )

        # Check output covariance and original input error model
        assert summary.error_structure is source_error
        assert summary.covariance is not None
        np.testing.assert_allclose(summary.covariance, [[0.5, 0.25], [0.25, 0.5]])


class TestNumericalOperatorPropagation:
    """Test module for nonlinear methods and repeatable errors when targets are reordered."""

    def test_propagate__nonlinear_operator_uses_numerical_fallback(self) -> None:
        """Checks that propagate() simulates nonlinear results and can omit their covariance."""

        # Small independent errors for nonlinear mean-square calculation
        data = _local([1, 3], ["a", "b"], [[0, 0], [1, 0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.2)])
        summary = gu.uncertainty.propagate(
            NonlinearMeanSquare(),
            data,
            structure,
            n_samples=2_000,
            random_state=7,
            return_covariance=False,
        )

        # Expected squared mean: 2**2 + Var(mean) = 4 + 0.2**2 / 2
        assert summary.method == "numerical"
        assert summary.estimate == pytest.approx(4)
        assert summary.mean == pytest.approx(4.02, abs=0.04)
        assert summary.covariance is None

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

    def test_propagate__error_samples_without_valid_sources(self) -> None:
        """Checks that a missing source cannot produce the requested random output samples."""

        # Missing source with no valid observation to sample
        data = LocalData(values=np.array([np.nan]), valid=np.array([False]), source_ids=np.array(["missing"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check sample request error for missing source
        with pytest.raises(ValueError, match="at least one valid source"):
            gu.uncertainty.propagate(Mean(), data, structure, return_samples=True)

    def test_propagate__masked_source_stays_invalid_in_numerical_draws(self) -> None:
        """Checks that a masked finite value is excluded from source errors and every simulated reduction."""

        # Masked finite value (first source only in error model)
        values = np.ma.array([2.0, 100.0], mask=[False, True])
        data = LocalData(
            values=values,
            valid=np.ones(2, dtype=bool),
            source_ids=np.asarray(["used", "masked"]),
        )
        labels = pd.Index(["used"])
        zero_covariance = pd.DataFrame([[0.0]], index=labels, columns=labels)
        errors = gu.ErrorStructure.from_gaussian(zero_covariance)

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


# Tests for intervals, marginals, and correlations in propagation results.


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

    def test_correlation__zero_variance_is_undefined(self) -> None:
        """Checks that a fixed result has zero covariance and no defined correlation with itself."""

        # Zero error for deterministic mean
        data = LocalData(values=np.array([1.0, 3.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])
        summary = gu.uncertainty.propagate(Mean(), data, structure, return_covariance=True)

        # Check zero covariance and undefined correlation (division by zero spread)
        assert summary.covariance is not None and summary.correlation is not None
        assert summary.covariance.loc["value", "value"] == 0
        assert np.isnan(summary.correlation.loc["value", "value"])

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

        # Fixed errors: mean shift (1 - 2) / 2, zero variance
        labels = pd.Index(["a", "b"])
        covariance = pd.DataFrame(np.zeros((2, 2)), index=labels, columns=labels)
        mean = pd.Series([1.0, -2.0], index=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance, mean=mean)
        data = LocalData(values=np.array([10.0, 20.0]), valid=np.ones(2, dtype=bool), source_ids=np.array(["a", "b"]))

        # Check shifted output mean and original source model
        summary = gu.uncertainty.propagate(Mean(), data, structure, method="numerical", n_samples=4)
        assert summary.error_structure is structure
        assert summary.estimate == pytest.approx(15)
        assert summary.mean == pytest.approx(14.5)
        assert summary.std == pytest.approx(0)
        assert not hasattr(summary, "to_error_structure")


class TestCallablePropagation:
    """Test module for complete callable calculations and selected spatial output distributions."""

    def test_propagate__same_source_value_cancels_in_complete_calculation(self) -> None:
        """Checks that repeated uses of one uncertain input share the same draw."""

        # Independent named inputs for repeated-use calculation
        labels = pd.Index(["a", "b"])
        covariance = pd.DataFrame(np.eye(2), index=labels, columns=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance)
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
            return_covariance=True,
        )

        # Check exact cancellation in difference, positive spread in sum
        assert summary.error_structure is structure
        assert summary.samples is not None and summary.covariance is not None
        np.testing.assert_array_equal(summary.samples["difference"], 0)
        assert summary.covariance.loc["difference", "difference"] == 0
        assert summary.std["sum"] > 0

    def test_propagate__large_raster_selects_only_requested_cell(self) -> None:
        """Checks that a large raster result saves draws only for explicitly selected cells."""

        # Zero error for exact selected value (row 3, column 4)
        values = np.arange(400, dtype=float).reshape(20, 20)
        raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 20), 32632)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 0)])

        # Select one output cell with small covariance limit
        summary = gu.uncertainty.propagate(
            lambda source: source,
            raster,
            structure,
            at=[(3, 4)],
            max_covariance_size=8,
            n_samples=2,
            return_samples=True,
            random_state=4,
        )

        # Check full raster mean and samples for selected cell only
        assert isinstance(summary.mean, gu.Raster)
        assert summary.samples is not None
        assert summary.samples.shape == (2, 1)
        np.testing.assert_array_equal(summary.samples[(3, 4)], [64, 64])

    def test_propagate__error_selected_samples_exceed_memory_limit(self) -> None:
        """Checks that sample storage is rejected before drawing when it exceeds the requested limit."""

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

    def test_propagate__error_local_options_with_callable(self) -> None:
        """Checks that options for local operators cannot silently alter a callable calculation."""

        # Whole-array callable with local-operator option
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        source = np.array([1.0, 2.0])

        # Check error for misplaced nodata option
        with pytest.raises(ValueError, match="apply only to local operators"):
            gu.uncertainty.propagate(
                lambda values: values.sum(),
                source,
                structure,
                nodata_propagation="ignore",
            )
