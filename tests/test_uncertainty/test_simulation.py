"""Tests for summaries calculated from repeated draws of source errors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import geoutils as gu
from geoutils.uncertainty.simulation import simulate


class TestSimulation:
    """Test module for partial draws, circular results, and failed calculations in simulate()."""

    def test_simulate__partial_draws_use_complete_cases_for_covariance(self) -> None:
        """Checks that each output uses its finite draws while covariance uses only complete pairs."""

        # Three draws, with the second output missing from the middle draw
        draws = iter((np.array([1.0, 2.0]), np.array([3.0, np.nan]), np.array([5.0, 6.0])))
        selection = pd.DataFrame({"flat_index": [0, 1], "estimate": [0.0, 0.0]}, index=["first", "second"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Save draws and quantiles while calculating covariance from the two complete pairs
        summary = simulate(
            estimate=np.zeros(2),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: draw,
            error_structure=structure,
            selection=selection,
            n_samples=3,
            return_covariance=True,
            return_samples=True,
            quantiles=(0.5,),
        )

        # Marginal counts are 3 and 2; complete pairs are [1, 2] and [5, 6]
        np.testing.assert_array_equal(summary.n_valid, [3, 2])
        np.testing.assert_allclose(summary.mean, [3, 4])
        np.testing.assert_allclose(summary.std, [2, np.sqrt(8)])
        assert summary.covariance_n == 2
        assert summary.covariance is not None
        np.testing.assert_allclose(summary.covariance, np.full((2, 2), 8.0))
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

        # Request covariance and quantile in the same unwrapped frame as the original angle
        summary = simulate(
            estimate=np.array(0.0),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: np.asarray(draw),
            error_structure=structure,
            selection=selection,
            n_samples=2,
            output={"circular_period": 360.0},
            return_covariance=True,
            return_samples=True,
            quantiles=(0.5,),
        )

        # Circular mean is zero; unwrapped selected values are -1 and 1 degrees
        assert summary.mean % 360 == pytest.approx(0, abs=1e-12)
        expected_resultant = np.cos(np.deg2rad(1))
        assert summary.resultant_length == pytest.approx(expected_resultant)
        assert summary.std == pytest.approx(180 / np.pi * np.sqrt(-2 * np.log(expected_resultant)))
        assert summary.covariance is not None
        assert summary.covariance.loc["angle", "angle"] == pytest.approx(2)
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

    def test_simulate__selected_output_has_no_complete_draws(self) -> None:
        """Checks that an always missing selected output has undefined covariance and mean."""

        # Each draw has one finite output, but the selected second output is always missing
        draws = iter((np.array([1.0, np.nan]), np.array([3.0, np.nan])))
        selection = pd.DataFrame({"flat_index": [1], "estimate": [np.nan]}, index=["missing"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check the finite first output and absent complete selected draws
        summary = simulate(
            estimate=np.array([0.0, np.nan]),
            draw_error=lambda _rng: next(draws),
            calculate=lambda draw: draw,
            error_structure=structure,
            selection=selection,
            n_samples=2,
            return_covariance=True,
        )
        np.testing.assert_allclose(summary.mean, [2, np.nan], equal_nan=True)
        assert summary.covariance_n == 0
        assert summary.covariance is not None and np.isnan(summary.covariance.iloc[0, 0])
        assert summary.covariance_mean is not None and np.isnan(summary.covariance_mean.iloc[0])

    @pytest.mark.parametrize(
        "bad_draw, message", [(np.array([1.0, 2.0]), "changed the output shape"), (np.nan, "no finite values")]
    )
    def test_simulate__error_bad_calculation_result(self, bad_draw: object, message: str) -> None:
        """Checks that draws with the wrong shape or no finite values fail before updating statistics."""

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
        """Checks that one successful draw cannot define a sample spread after skipped failures."""

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

    def test_simulate__error_fatal_calculation(self) -> None:
        """Checks that a declared fatal error is raised even when ordinary failed draws are skipped."""

        # Fatal errors indicate an invalid calculation rather than one unusable random draw
        selection = pd.DataFrame({"flat_index": [0], "estimate": [0.0]}, index=["value"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        def calculate(_draw: float) -> float:
            """Raise the declared fatal error on every draw."""

            raise TypeError("invalid calculation")

        # Check that warn mode does not swallow the contract error
        with pytest.raises(TypeError, match="invalid calculation"):
            simulate(
                estimate=np.array(0.0),
                draw_error=lambda _rng: 1.0,
                calculate=calculate,
                error_structure=structure,
                selection=selection,
                n_samples=2,
                on_error="warn",
                fatal_exceptions=(TypeError,),
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
        """Checks that invalid draw counts, probabilities, and storage limits fail before sampling."""

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
        """Checks that selected results have unique keys and positions inside the output."""

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
