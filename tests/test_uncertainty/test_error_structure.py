"""Tests for error structure classes."""

from __future__ import annotations

import pickle
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
import pytest

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.stats.variography import VariogramModel


class TestErrorMagnitude:
    """Test module for the ErrorMagnitude class."""

    def test_constant__fixed_magnitude(self) -> None:
        """Checks that a constant magnitude predicts its fixed standard deviation."""

        # Constant STD
        magnitude = gu.ErrorMagnitude.constant(2)

        # Check attributes/prediction
        assert magnitude.kind == "constant"
        assert magnitude.predict() == 2
        assert magnitude.reference_value == 2

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("statistic", ["nmad", "std"])
    def test_variable_from_grouped_stats__optional_value_name(self, named_columns: bool, statistic: str) -> None:
        """Checks that grouped stats input can be passed with/without named value columns, and both work."""

        # We define magnitude that increase from 1 to 3 between slopes of 0 and 30 degrees
        statistics = pd.DataFrame({statistic: [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        # We add a MultiIndex name to test multi-level (default output dataframe with named values in stats())
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])
        original = statistics.copy(deep=True)

        # We predict values inside/outside the slope categories
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        predicted = magnitude.predict({"slope": [-5.0, 0.0, 15.0, 30.0, 35.0]})

        # We check the prediction is interpolating linearly between 0/30 degrees, using nearest beyond
        np.testing.assert_array_equal(predicted, [1.0, 1.0, 2.0, 3.0, 3.0])
        assert magnitude.reference_value == 2.0
        assert magnitude.statistic == statistic
        assert magnitude.kind == "variable"
        pd.testing.assert_frame_equal(statistics, original)

    @pytest.mark.parametrize("statistic", ["nmad", "std"])
    def test_variable_from_grouped_stats__auto_stat(self, statistic: str) -> None:
        """Checks that we automatically select STD or NMAD (if no user input), otherwise user input."""

        # Use STD/NMAD on two different variables
        other_statistic = "std" if statistic == "nmad" else "nmad"
        statistics = pd.DataFrame(
            {
                ("dz", statistic): [1.0, 3.0],
                ("dz", "count"): [20, 30],
                ("other", other_statistic): [10.0, 30.0],
                ("other", "count"): [20, 30],
            },
            index=pd.Index([0.0, 30.0], name="slope"),
        )

        # Only one statistic is available for dz, so there should be no ambiguity warning
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, value_name="dz")

        # We check dz uses its associated spread statistic, regardless of the other
        assert magnitude.statistic == statistic
        assert magnitude.predict({"slope": 15.0}) == 2.0

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("direct_constructor", [False, True])
    def test_variable_from_grouped_stats__multiple_predictors(
        self, named_columns: bool, direct_constructor: bool
    ) -> None:
        """
        Checks that, whether built directly or through variable_from_grouped_stats(), multiple predictors work as
        intended for prediction
        ."""

        # We use 4 groups, defining a plane with midpoint of 2.5 to test further below
        index = pd.MultiIndex.from_product([[0.0, 30.0], [100.0, 1000.0]], names=["slope", "elevation"])
        statistics = pd.DataFrame({"std": [1.0, 2.0, 3.0, 4.0], "count": [20, 20, 20, 20]}, index=index)
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["dz"], statistics.columns])

        # Direct construction or variable_from_grouped_stats()
        if direct_constructor:
            magnitude = gu.ErrorMagnitude(
                kind="variable",
                grouped_statistics=statistics,
                predictor_names=("slope", "elevation"),
                value_name="dz",
            )
        else:
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, value_name="dz")
        predicted = magnitude.predict({"slope": [0.0, 15.0, 30.0], "elevation": [100.0, 550.0, 1000.0]})

        # Check predictions match the corner + midpoint
        np.testing.assert_array_equal(predicted, [1.0, 2.5, 4.0])
        assert magnitude.reference_value == 2.5
        assert magnitude.value is None
        assert magnitude.kind == "variable"

    def test_variable_from_grouped_stats__fills_missing_corner(self) -> None:
        """Checks that a missing (NaN) group uses nearest neighbor before interpolation."""

        # We define 2x2 bins with an empty one
        index = pd.MultiIndex.from_product([[0.0, 3.0], [0.0, 10.0]], names=["slope", "elevation"])
        statistics = pd.DataFrame({"std": [1.0, 2.0, 4.0, np.nan], "count": [20] * 4}, index=index)

        # We check the missing corner is filling with nearest neighbour
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        missing_corner = magnitude.predict({"slope": 3.0, "elevation": 10.0})
        midpoint = magnitude.predict({"slope": 1.5, "elevation": 5.0})
        assert missing_corner == 2
        assert midpoint == np.mean([1.0, 2.0, 4.0, 2.0])

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("statistic, expected", [("nmad", 2.0), ("std", 4.0), ("mad", 6.0)])
    def test_variable_from_grouped_stats__explicit_statistic(
        self, named_columns: bool, statistic: str, expected: float
    ) -> None:
        """Checks that a user-input statistic is respected."""

        # We use STD/NMAD/MAD in the same dataframe
        statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "std": [2.0, 6.0], "mad": [3.0, 9.0], "count": [20, 30]},
            index=pd.Index([0.0, 30.0], name="slope"),
        )
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # We select which one we use, which should NOT trigger a warning (otherwise this test fails)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, statistic=statistic)

        # We check prediction and reference value equal the chosen stat midpoint
        assert magnitude.predict({"slope": 15.0}) == expected
        assert magnitude.reference_value == expected
        assert magnitude.statistic == statistic

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("direct_constructor", [False, True])
    def test_variable_from_grouped_stats__warns_ambiguous_statistic(
        self, named_columns: bool, direct_constructor: bool
    ) -> None:
        """Checks that automatic selection warns and uses STD alone when both STD and NMAD are present."""

        # We define both NMAD/STD
        statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "std": [2.0, 6.0], "count": [20, 30]},
            index=pd.Index([0.0, 30.0], name="slope"),
        )
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # Check warning is raised about ambiguity + choice of STD per default
        with pytest.warns(UserWarning, match="Both 'std' and 'nmad'.*using 'std'"):
            if direct_constructor:
                magnitude = gu.ErrorMagnitude(
                    kind="variable", grouped_statistics=statistics, predictor_names=("slope",)
                )
            else:
                magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

        # The values here should match the STD only
        np.testing.assert_array_equal(magnitude.predict({"slope": [0.0, 15.0, 30.0]}), [2.0, 4.0, 6.0])
        assert magnitude.reference_value == 4.0
        assert magnitude.statistic == "std"

    @pytest.mark.parametrize("named_columns", [False, True])
    def test_variable_from_grouped_stats__min_count(self, named_columns: bool) -> None:
        """Checks that groups with too few sample counts are properly excluded."""

        # We only define slopes 0 and 30 with finite/positive values and counts above 10
        statistics = pd.DataFrame(
            {"nmad": [1.0, -2.0, 100.0, 3.0, np.nan], "count": [20, 20, 1, 20, 20]},
            index=pd.Index([0.0, 10.0, 20.0, 30.0, 40.0], name="slope"),
        )
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])
        original = statistics.copy(deep=True)

        # We use min_count=10, then scale/offset the input
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(
            statistics, min_count=10, scale=2, variance_offset=4, floor=0.5
        )
        predicted = magnitude.predict({"slope": [0.0, 15.0, 30.0]})

        # We check the calculation happened after exclusion by min count, i.e. only using NMADS from slopes 0/30
        # Magnitudes 2, 4 and 6 give variances 0, 12 and 32 + floor of 0.5 for the first one
        expected = [0.5, np.sqrt(12), np.sqrt(32)]
        assert np.allclose(predicted, expected, equal_nan=True)
        assert magnitude.reference_value == pytest.approx((0.5 + np.sqrt(32)) / 2)
        pd.testing.assert_frame_equal(statistics, original)

    def test_pickle__magnitude(self) -> None:
        """Checks that an ErrorMagnitude can be pickled."""

        # We create a synthetic error magnitude
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [20, 20]}, index=pd.Index([0.0, 30.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

        # Try to restore it from pickled
        restored = pickle.loads(pickle.dumps(magnitude))

        # Check equality
        pd.testing.assert_frame_equal(restored.grouped_statistics, magnitude.grouped_statistics)
        np.testing.assert_array_equal(restored.predict({"slope": [0.0, 15.0, 30.0]}), [1.0, 2.0, 3.0])


class TestErrorMagnitudeErrors:
    """Test module for errors raised by ErrorMagnitude."""

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("missing_column", ["nmad", "count"])
    def test_variable_from_grouped_stats__error_missing_column(self, named_columns: bool, missing_column: str) -> None:
        """Checks error is raised when either column lacks a spread statistic or sample counts."""

        # We remove one required column
        statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope")
        ).drop(columns=missing_column)
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        with pytest.raises(ValueError, match=f"must contain.*{missing_column}"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

    @pytest.mark.parametrize("named_columns", [False, True])
    def test_variable_from_grouped_stats__error_statistic_requires_explicit_choice(self, named_columns: bool) -> None:
        """Checks error is raised when a custom spread statistic is not selected explicitly."""

        # MAD is not recognized automatically
        statistics = pd.DataFrame({"mad": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        with pytest.raises(ValueError, match="pass statistic explicitly"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

    @pytest.mark.parametrize("named_columns", [False, True])
    def test_variable_from_grouped_stats__error_missing_requested_statistic(self, named_columns: bool) -> None:
        """Checks error is raised when the requested statistic is missing from the table."""

        # The requested STD is absent (even though nmad could be selected automatically)
        statistics = pd.DataFrame({"nmad": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])
        with pytest.raises(ValueError, match="must contain.*std"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics, statistic="std")

    def test_predict__error_missing_predictor(self) -> None:
        """Checks error is raised when a variable magnitude lacks its named predictor."""

        # We used a slope predictor, so need to pass it again during prediction
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [20, 20]}, index=pd.Index([0.0, 30.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

        with pytest.raises(ValueError, match="Missing predictors.*slope"):
            magnitude.predict({"elevation": 100.0})

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"value": -1.0}, "finite, non-negative value"),
            ({"scale": 0.0}, "scale must be finite and strictly positive"),
            ({"variance_offset": -1.0}, "variance_offset must be finite and non-negative"),
            ({"floor": -1.0}, "floor must be finite and non-negative"),
        ],
    )
    def test_init__error_invalid_constant_magnitude(self, options: dict[str, float], message: str) -> None:
        """Checks error is raised for invalid magnitudes."""

        # We use a valid constant dictionary, and update it with invalid test parameter one at a time
        parameters: dict[str, Any] = {"kind": "constant", "value": 1.0} | options
        with pytest.raises(ValueError, match=message):
            gu.ErrorMagnitude(**parameters)


class TestErrorComponent:
    """Test module for ErrorComponent class."""

    def test_init__independent_component(self) -> None:
        """Checks basic creation without correlation."""

        # Name and STD value
        component = gu.ErrorComponent("measurement", 2)

        # Check argument
        assert component.name == "measurement"
        assert isinstance(component.magnitude, gu.ErrorMagnitude)
        assert component.predict_magnitude() == 2
        np.testing.assert_array_equal(component.predict_correlation([0, 10]), [1, 0])

    def test_init__uses_fitted_variogram_correlation(self) -> None:
        """Checks creation with variogram for correlation."""

        pytest.importorskip("skgstat")

        # Synthetic variogram, with partial sill not normalized to 1
        model = VariogramModel("spherical", effective_range=10, partial_sill=4)
        variogram = gu.Variogram(
            lags=np.array([0.0, 10.0]), semivariance=np.array([0.0, 4.0]), counts=np.array([5, 5]), model=model
        )
        component = gu.ErrorComponent("spatial", 2, variogram)

        # We check the variogram is used for correlation, normalized to 1
        assert isinstance(component.correlation, VariogramModel)
        assert component.correlation.sill == 1
        assert component.predict_magnitude() == 2
        np.testing.assert_array_equal(component.predict_correlation([0.0, 10.0]), [1.0, 0.0])

    def test_pickle__component(self) -> None:
        """Checks that an ErrorComponent can be pickled."""

        pytest.importorskip("skgstat")

        # We create a synthetic error component
        correlation = VariogramModel("spherical", effective_range=10, partial_sill=1)
        component = gu.ErrorComponent("spatial", 2.0, correlation, metadata={"source": "sample"})

        # We try to restore the component from its pickle
        restored = pickle.loads(pickle.dumps(component))

        # We check equality and read-only metadata
        assert restored.predict_magnitude() == 2.0
        np.testing.assert_array_equal(restored.predict_correlation([0.0, 10.0]), [1.0, 0.0])
        assert dict(restored.metadata) == {"source": "sample"}
        with pytest.raises(TypeError):
            restored.metadata["source"] = "changed"


class TestErrorComponentErrors:
    """Test module for errors raised by ErrorComponent."""

    @pytest.mark.parametrize(
        "name, magnitude, correlation, error_type, message",
        [
            ("", 2, None, ValueError, "non-empty string"),
            ("measurement", "large", None, TypeError, "must be numeric"),
            ("spatial", 2, "spherical", TypeError, "VariogramModel or Variogram"),
        ],
    )
    def test_init__error_invalid_component(
        self, name: str, magnitude: object, correlation: object, error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for an invalid component name, magnitude, or correlation."""

        with pytest.raises(error_type, match=message):
            gu.ErrorComponent(name, magnitude, correlation)  # type: ignore[arg-type]


class TestErrorStructure:
    """Test module for ErrorStructure class."""

    def test_init__single_component(self) -> None:
        """Checks basic creation with single component."""

        component = gu.ErrorComponent("measurement", 2)
        structure = gu.ErrorStructure([component])

        # Check attributes
        assert list(structure.components) == ["measurement"]
        assert structure.predict_magnitude() == 2
        assert structure.predict_variance() == 4

    def test_to_covariance_matrix__coincident_independence(self) -> None:
        """Checks that observations at the same coordinates have independent measurement errors."""

        # We create coincident coordinates
        coordinates = np.array([[1.0, 2.0], [1.0, 2.0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Check diagonal covariance is independent
        covariance = structure.to_covariance_matrix(coordinates)
        np.testing.assert_array_equal(covariance, np.diag([4.0, 4.0]))

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_components__combine_magnitude_and_correlation(self) -> None:
        """Checks that component variances add while variogram amplitudes are normalized."""

        # We combine correlated and independent synthetic components
        spatial = gu.ErrorComponent("spatial", 2, VariogramModel("gaussian", 10, 3))
        independent = gu.ErrorComponent("independent", 0.5)
        structure = gu.ErrorStructure([spatial, independent])

        # Normalize the spatial sill from 3 to 1, then add variances 2**2 + 0.5**2 = 4.25
        # At distance 0, both correlations are 1, so (4 * 1 + 0.25 * 1) / 4.25 = 1
        assert isinstance(spatial.correlation, VariogramModel) and spatial.correlation.sill == 1
        assert structure.predict_variance() == pytest.approx(4.25)
        assert structure.predict_correlation(0) == pytest.approx(1)

        # Compare covariance matrix with pairwise calculation
        coordinates = np.array([[0.0, 0.0], [10.0, 0.0]])
        covariance = structure.to_covariance_matrix(coordinates)
        np.testing.assert_allclose(np.diag(covariance), [4.25, 4.25])
        assert covariance[0, 1] == pytest.approx(structure.predict_covariance(10))

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_to_covariance_matrix__uses_selected_coordinate_dimensions(self) -> None:
        """Checks that covariance uses only the dimensions selected by a spatial component."""

        # Coincident X coordinates with different Y
        coordinates = np.array([[0.0, 0.0], [0.0, 10.0], [1.0, 0.0]])
        correlation = VariogramModel("exponential", effective_range=4, partial_sill=1, active_dims=(0,))
        structure = gu.ErrorStructure([gu.ErrorComponent("horizontal", 2, correlation)])

        # Check covariance based on X separation only
        covariance = structure.to_covariance_matrix(coordinates)
        assert covariance[0, 1] == pytest.approx(4)
        assert covariance[0, 2] == pytest.approx(4 * correlation.correlation(1))
        assert covariance[0, 2] < covariance[0, 1]

    def test_variable_magnitude__preserves_total_local_variance(self) -> None:
        """Checks that a variable component subtracts variance assigned to a fixed component."""

        # Grouped total magnitudes, split into variable and fixed components
        intervals = pd.IntervalIndex.from_breaks([0, 1, 2], name="slope")
        columns = pd.MultiIndex.from_tuples([("error", "nmad"), ("error", "count")])
        statistics = pd.DataFrame([[1.0, 100], [2.0, 100]], index=intervals, columns=columns)
        variable = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, variance_offset=0.25)
        structure = gu.ErrorStructure([gu.ErrorComponent("variable", variable), gu.ErrorComponent("fixed", 0.5)])

        # Check original total magnitude at group centers
        predictors = {"slope": np.array([0.5, 1.5])}
        np.testing.assert_allclose(structure.predict_magnitude(predictors), [1.0, 2.0])
        covariance = structure.to_covariance_matrix(np.array([[0.0, 0.0], [1.0, 0.0]]), predictors=predictors)
        np.testing.assert_allclose(np.diag(covariance), [1.0, 4.0])

    def test_pickle__error_structure(self) -> None:
        """Checks that an ErrorStructure can be pickled."""

        # We create a synthetic variable and constant components with an empirical variogram and fit details
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        correlation = VariogramModel("spherical", effective_range=10, partial_sill=1)
        empirical = gu.Variogram(
            lags=np.array([0.0, 10.0]), semivariance=np.array([0.0, 1.0]), counts=np.array([5, 5]), model=correlation
        )
        structure = gu.ErrorStructure(
            [
                gu.ErrorComponent("terrain", magnitude, correlation, metadata={"group": "slope"}),
                gu.ErrorComponent("noise", 0.5),
            ],
            empirical_variogram=empirical,
            fit_diagnostics={"sample_count": 40},
            metadata={"source": "example"},
        )

        # We try to restore the complete model from its pickle
        restored = pickle.loads(pickle.dumps(structure))

        # We check equality and order
        assert list(restored.components) == ["terrain", "noise"]
        np.testing.assert_allclose(
            restored.predict_magnitude({"slope": [0.0, 15.0, 30.0]}), np.sqrt([1.25, 4.25, 9.25])
        )
        assert restored.components["terrain"].metadata == {"group": "slope"}
        assert restored.fit_diagnostics == {"sample_count": 40}
        assert restored.metadata == {"source": "example"}
        assert restored.empirical_variogram is not None
        np.testing.assert_array_equal(restored.empirical_variogram.semivariance, [0.0, 1.0])
        with pytest.raises(TypeError):
            restored.components["extra"] = gu.ErrorComponent("extra", 1.0)

    def test_iter_samples__adds_errors_to_source_values(self) -> None:
        """Checks that value samples add the same sampled errors to each source value."""

        # Independent unit errors for two named sources
        source_ids = pd.Index(["a", "b"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        nominal = np.array([10.0, 20.0])

        # Check matching error draws and source values plus errors
        error_samples = list(structure.iter_samples(source_ids=source_ids, n_samples=2, random_state=4))
        value_samples = list(
            structure.iter_samples(source_ids=source_ids, nominal=nominal, kind="value", n_samples=2, random_state=4)
        )
        errors = np.asarray(error_samples)
        values = np.asarray(value_samples)
        assert errors.shape == (2, 2)
        np.testing.assert_array_equal(values, nominal + errors)

    def test_info__variable_magnitude(self) -> None:
        """Checks that info() describes a variable magnitude by the predictors that control it."""

        # The error magnitude varies with slope, while observations remain independent
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)])

        # The summary names the predictor instead of describing the table format
        summary = structure.info(verbose=False)
        assert summary is not None
        assert "terrain: varies with slope, independent" in summary

    @pytest.mark.skipif(find_spec("matplotlib") is None, reason="Requires Matplotlib")
    def test_plot_correlation(self) -> None:
        """Checks that correlation plot runs and contains the right axes/data."""

        pytest.importorskip("skgstat")
        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        # We create a synthetic error structure
        model = VariogramModel("gaussian", effective_range=3, partial_sill=1)
        empirical = gu.Variogram(
            lags=np.array([1.0, 2.0]), semivariance=np.array([0.2, 0.7]), counts=np.array([8, 9]), model=model
        )
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, model)], empirical_variogram=empirical)

        # Both panels should contain correlation axes
        axes = structure.plot_correlation()
        panels = structure.plot()
        try:
            assert set(panels) == {"correlation"}
            assert axes.get_xlabel() == "Lag distance"
            assert panels["correlation"].get_ylabel() == "Semivariance"
        finally:
            plt.close(axes.figure)
            plt.close(panels["correlation"].figure)

    @pytest.mark.skipif(find_spec("matplotlib") is None, reason="Requires Matplotlib")
    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("method", ["plot_magnitude", "plot"])
    def test_plot__variable_magnitude(self, named_columns: bool, method: str) -> None:
        """Checks plotting for variable error magnitude, through plot() or plot_magnitude()."""

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        # We define 2 slope bins
        index = pd.IntervalIndex.from_breaks([0.0, 10.0, 20.0], name="slope")
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [20, 30]}, index=index)
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product(
                [["error"], statistics.columns], names=["value", "statistic"]
            )

        # We define a structure with a constant and a variable magnitude component, only the second can be plotted
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude), gu.ErrorComponent("noise", 0.5)])
        panels = getattr(structure, method)()
        if method == "plot":
            assert set(panels) == {"magnitude:terrain"}
            panels = panels["magnitude:terrain"]

        # We check the curve uses bin centers, and the count bars show the observations behind each estimate
        try:
            np.testing.assert_array_equal(panels["statistic"].lines[0].get_xdata(), [5.0, 15.0])
            np.testing.assert_array_equal(panels["statistic"].lines[0].get_ydata(), [1.0, 3.0])
            assert [bar.get_height() for bar in panels["count"].patches] == [20, 30]
        finally:
            plt.close(panels["statistic"].figure)


class TestErrorStructureErrors:
    """Test module for errors/warnings of the ErrorStructure class."""

    def test_plot__warns_without_empirical_diagnostics(self) -> None:
        """Checks that a plot without empirical variogram warns."""

        # Synthetic error structure without empirical vario
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.warns(UserWarning, match="no empirical diagnostics"):
            assert structure.plot() == {}
        with pytest.raises(ValueError, match="No empirical variogram"):
            structure.plot_correlation()

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"source_ids": ["a", "b"], "n_samples": 0}, "n_samples must be a positive integer"),
            ({"source_ids": ["a", "b"], "kind": "unknown"}, "kind must"),
            ({"source_ids": ["a", "b"], "nominal": [1.0]}, "nominal must contain one value"),
        ],
    )
    def test_iter_samples__error_invalid_request(self, options: dict[str, object], message: str) -> None:
        """Checks an error is raised for an invalid sample count, kind, or source values."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match=message):
            list(structure.iter_samples(**options))  # type: ignore[arg-type]
