"""Tests for reusable error components and labelled Gaussian errors."""

from __future__ import annotations

import warnings
from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.stats.variography import VariogramModel


class TestErrorMagnitude:
    """Test module for variable magnitudes with optional variable names and multiple predictors."""

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("statistic", ["nmad", "std"])
    def test_variable_from_grouped_stats__optional_value_name(self, named_columns: bool, statistic: str) -> None:
        """Checks that plain and two-level statistic columns give the same interpolated magnitudes."""

        # Magnitudes increase from 1 to 3 between slopes of 0 and 30 degrees
        statistics = pd.DataFrame({statistic: [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])
        original = statistics.copy(deep=True)

        # Interpolate within the sampled slopes and use the nearest group beyond them
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        predicted = magnitude.predict({"slope": [-5.0, 0.0, 15.0, 30.0, 35.0]})

        # The midpoint and median are 2, and construction leaves the input table unchanged
        np.testing.assert_array_equal(predicted, [1.0, 1.0, 2.0, 3.0, 3.0])
        assert magnitude.reference_value == 2.0
        assert magnitude.statistic == statistic
        assert magnitude.kind == "variable"
        pd.testing.assert_frame_equal(statistics, original)

    @pytest.mark.parametrize("statistic", ["nmad", "std"])
    def test_variable_from_grouped_stats__automatic_statistic_uses_selected_variable(self, statistic: str) -> None:
        """Checks that automatic selection ignores spread statistics belonging to other variables."""

        # The selected variable has one spread statistic, while another variable has the other
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
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, value_name="dz")

        # The midpoint uses the dz magnitudes, regardless of the other variable's statistic
        assert magnitude.statistic == statistic
        assert magnitude.predict({"slope": 15.0}) == 2.0

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("direct_constructor", [False, True])
    def test_variable_from_grouped_stats__multiple_predictors(
        self, named_columns: bool, direct_constructor: bool
    ) -> None:
        """Checks that both constructors interpolate standard deviations over slope and elevation."""

        # Four group centers define a plane with a known midpoint magnitude of 2.5
        index = pd.MultiIndex.from_product([[0.0, 30.0], [100.0, 1000.0]], names=["slope", "elevation"])
        statistics = pd.DataFrame({"std": [1.0, 2.0, 3.0, 4.0], "count": [20, 20, 20, 20]}, index=index)
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["dz"], statistics.columns])

        # Direct construction supplies predictor names; variable_from_grouped_stats() infers them from the index
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

        # Predictions match the corner values and their average at the midpoint
        np.testing.assert_array_equal(predicted, [1.0, 2.5, 4.0])
        assert magnitude.reference_value == 2.5
        assert magnitude.value is None
        assert magnitude.kind == "variable"


class TestErrorMagnitudeOptions:
    """Test module for filtering and scaling variable magnitudes with either column format."""

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("direct_constructor", [False, True])
    def test_variable_from_grouped_stats__warns_for_ambiguous_statistic(
        self, named_columns: bool, direct_constructor: bool
    ) -> None:
        """Checks that automatic selection warns and uses std alone when both std and nmad are present."""

        # Different spread estimates reveal whether std is selected or combined with nmad
        statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "std": [2.0, 6.0], "count": [20, 30]},
            index=pd.Index([0.0, 30.0], name="slope"),
        )
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # Both construction methods report the same default choice
        with pytest.warns(UserWarning, match="Both 'std' and 'nmad'.*using 'std'"):
            if direct_constructor:
                magnitude = gu.ErrorMagnitude(
                    kind="variable", grouped_statistics=statistics, predictor_names=("slope",)
                )
            else:
                magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

        # Interpolation and the reference value use only the standard deviations
        np.testing.assert_array_equal(magnitude.predict({"slope": [0.0, 15.0, 30.0]}), [2.0, 4.0, 6.0])
        assert magnitude.reference_value == 4.0
        assert magnitude.statistic == "std"

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("statistic, expected", [("nmad", 2.0), ("std", 4.0), ("mad", 6.0)])
    def test_variable_from_grouped_stats__explicit_statistic(
        self, named_columns: bool, statistic: str, expected: float
    ) -> None:
        """Checks that an explicit statistic selects its own estimates without an ambiguity warning."""

        # Each statistic has a different midpoint, including a spread that is never selected automatically
        statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "std": [2.0, 6.0], "mad": [3.0, 9.0], "count": [20, 30]},
            index=pd.Index([0.0, 30.0], name="slope"),
        )
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # An explicit choice removes ambiguity even though several spread estimates are present
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, statistic=statistic)

        # Both prediction and the reference value equal the chosen estimates' midpoint
        assert magnitude.predict({"slope": 15.0}) == expected
        assert magnitude.reference_value == expected
        assert magnitude.statistic == statistic

    @pytest.mark.parametrize("named_columns", [False, True])
    def test_variable_from_grouped_stats__count_threshold_and_variance_offset(self, named_columns: bool) -> None:
        """Checks that invalid groups are excluded before scaling magnitudes and subtracting component variance."""

        # Only slopes 0 and 30 have non-negative, finite magnitudes based on enough observations
        statistics = pd.DataFrame(
            {"nmad": [1.0, -2.0, 100.0, 3.0, np.nan], "count": [20, 20, 1, 20, 20]},
            index=pd.Index([0.0, 10.0, 20.0, 30.0, 40.0], name="slope"),
        )
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])
        original = statistics.copy(deep=True)

        # Double the magnitudes and subtract variance already assigned to another component
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(
            statistics, min_count=10, scale=2, variance_offset=4, floor=0.5
        )
        predicted = magnitude.predict({"slope": [0.0, 15.0, 30.0]})

        # Magnitudes 2, 4 and 6 give variances 0, 12 and 32; the floor raises the first magnitude to 0.5
        expected = [0.5, np.sqrt(12), np.sqrt(32)]
        np.testing.assert_allclose(predicted, expected, rtol=1e-14)
        assert magnitude.reference_value == pytest.approx((0.5 + np.sqrt(32)) / 2)
        pd.testing.assert_frame_equal(statistics, original)


class TestErrorStructureComponents:
    """Test module for component magnitudes, spatial correlation, and observation identity."""

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_components__combine_magnitude_and_correlation(self) -> None:
        """Checks that component variances add while variogram amplitudes are normalized."""

        # Combine correlated and independent components
        spatial = gu.ErrorComponent("spatial", 2, VariogramModel("gaussian", 10, 3))
        independent = gu.ErrorComponent("independent", 0.5)
        structure = gu.ErrorStructure([spatial, independent])

        # Check total variance and normalized correlation
        assert isinstance(spatial.correlation, VariogramModel) and spatial.correlation.sill == 1
        assert structure.predict_variance() == pytest.approx(4.25)
        assert structure.predict_correlation(0) == pytest.approx(1)

        # Compare covariance matrix with pairwise calculation
        coordinates = np.array([[0.0, 0.0], [10.0, 0.0]])
        covariance = structure.to_covariance_matrix(coordinates)
        np.testing.assert_allclose(np.diag(covariance), [4.25, 4.25])
        assert covariance[0, 1] == pytest.approx(structure.predict_covariance(10))

    def test_to_covariance_matrix__coincident_observations_stay_independent(self) -> None:
        """Checks that observations at the same coordinates still have independent measurement errors."""

        # Independent errors at coincident coordinates
        coordinates = np.array([[1.0, 2.0], [1.0, 2.0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Check diagonal covariance for distinct observations
        covariance = structure.to_covariance_matrix(coordinates)
        np.testing.assert_array_equal(covariance, np.diag([4.0, 4.0]))

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


class TestErrorStructureGaussian:
    """Test module for labelled Gaussian validation and singular random realizations."""

    def test_from_gaussian__preserves_singular_draws(self) -> None:
        """Checks that perfect dependence and deterministic parameters gain no artificial noise."""

        # Perfectly correlated pair plus fixed error
        labels = pd.Index(["a", "b", "fixed"])
        covariance = pd.DataFrame([[4.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], index=labels, columns=labels)
        mean = pd.Series([1.0, -1.0, 3.0], index=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance, mean=mean, units="m")

        # Check exact dependence and fixed third error
        fields = structure.generate_random_field(n_fields=4, random_state=4)
        assert fields.shape == (4, 3)
        np.testing.assert_allclose(fields[:, 0] - 1, 2 * (fields[:, 1] + 1), atol=1e-14)
        np.testing.assert_allclose(fields[:, 2], 3)

    def test_iter_samples__adds_gaussian_errors_to_source_values(self) -> None:
        """Checks that value samples add labelled Gaussian mean errors to the matching source values."""

        # Zero covariance for deterministic mean errors
        labels = pd.Index(["a", "b"])
        covariance = pd.DataFrame(np.zeros((2, 2)), index=labels, columns=labels)
        mean = pd.Series([1.0, -2.0], index=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance, mean=mean)

        # Check fixed error draws and source values plus errors
        errors = list(structure.iter_samples(n_samples=2, random_state=4))
        values = list(structure.iter_samples(nominal=[10, 20], kind="value", n_samples=2, random_state=4))
        np.testing.assert_array_equal(errors, [[1.0, -2.0], [1.0, -2.0]])
        np.testing.assert_array_equal(values, [[11.0, 18.0], [11.0, 18.0]])


class TestErrorStructureBinding:
    """Test module for matching reusable error components to named source observations."""

    def test_bind__tuple_source_ids_remain_distinct(self) -> None:
        """Checks that tuple source IDs identify two independent observations in a bound model."""

        # Tuple IDs for two independent observations
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        source_ids = [(1, "a"), (2, "b")]

        # Check diagonal covariance with each tuple treated as one ID
        bound = structure.bind(source_ids)
        covariance = bound.covariance_block([0, 1], [0, 1])
        np.testing.assert_array_equal(covariance, np.diag([4.0, 4.0]))


class TestErrorStructureDiagnostics:
    """Test module for variable magnitude summaries and grouped diagnostic plots."""

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

    def test_info__gaussian_labels(self) -> None:
        """Checks that a Gaussian error summary names its stored source observations."""

        # Fixed mean and covariance for two labelled observations
        labels = pd.Index(["left", "right"])
        covariance = pd.DataFrame(np.eye(2), index=labels, columns=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance, mean=pd.Series([1.0, 2.0], index=labels))

        # The summary describes the joint vector and its labels
        summary = structure.info(verbose=False)
        assert summary is not None
        assert "joint Gaussian vector of 2 parameter(s)" in summary
        assert "labels: left, right" in summary

    @pytest.mark.skipif(find_spec("matplotlib") is None, reason="Requires Matplotlib")
    def test_plot_correlation(self) -> None:
        """Checks that correlation plot runs and contains the right axes/data."""

        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        # We create a synthetic error structure
        model = VariogramModel("gaussian", effective_range=3, partial_sill=1)
        empirical = gu.Variogram(
            lags=np.array([1.0, 2.0]), semivariance=np.array([0.2, 0.7]), counts=np.array([8, 9]), model=model
        )
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("spatial", 1, model)], empirical_variogram=empirical
        )

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

    def test_plot__warns_without_empirical_diagnostics(self) -> None:
        """Checks that a plot without empirical variogram warns."""

        # Synthetic error structure without empirical vario
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.warns(UserWarning, match="no empirical diagnostics"):
            assert structure.plot() == {}
        with pytest.raises(ValueError, match="No empirical variogram"):
            structure.plot_correlation()

    @pytest.mark.skipif(find_spec("matplotlib") is None, reason="Requires Matplotlib")
    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("method", ["plot_magnitude", "plot"])
    def test_methods__variable_magnitude_plotting(self, named_columns: bool, method: str) -> None:
        """Checks that both plotting methods select the variable component and show its magnitudes and counts."""

        # Two slope bins have known magnitudes at their centers of 5 and 15 degrees
        import_optional("matplotlib")
        import matplotlib.pyplot as plt

        index = pd.IntervalIndex.from_breaks([0.0, 10.0, 20.0], name="slope")
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [20, 30]}, index=index)
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product(
                [["error"], statistics.columns], names=["value", "statistic"]
            )

        # Only the variable component has grouped statistics to plot
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude), gu.ErrorComponent("noise", 0.5)])
        panels = getattr(structure, method)()
        if method == "plot":
            assert set(panels) == {"magnitude:terrain"}
            panels = panels["magnitude:terrain"]

        # The curve uses bin centers, and the count bars show the observations behind each estimate
        try:
            np.testing.assert_array_equal(panels["statistic"].lines[0].get_xdata(), [5.0, 15.0])
            np.testing.assert_array_equal(panels["statistic"].lines[0].get_ydata(), [1.0, 3.0])
            assert [bar.get_height() for bar in panels["count"].patches] == [20, 30]
        finally:
            plt.close(panels["statistic"].figure)


class TestErrorStructureGaussianErrors:
    """Test module for invalid labelled Gaussian covariance matrices."""

    def test_from_gaussian__error(self) -> None:
        """Checks invalid labels/covariance."""

        labels = pd.Index(["x", "y"])
        valid = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=labels, columns=labels)
        with pytest.raises(ValueError, match="match exactly"):
            gu.ErrorStructure.from_gaussian(valid.rename(columns={"y": "z"}))
        with pytest.raises(ValueError, match="zero-variance"):
            gu.ErrorStructure.from_gaussian(pd.DataFrame([[0.0, 0.1], [0.1, 1.0]], index=labels, columns=labels))
        with pytest.raises(ValueError, match="positive semidefinite"):
            gu.ErrorStructure.from_gaussian(pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=labels, columns=labels))


class TestErrorMagnitudeValidation:
    """Test module for required statistic and count columns in grouped magnitude tables."""

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("missing_column", ["nmad", "count"])
    def test_variable_from_grouped_stats__error_missing_column(self, named_columns: bool, missing_column: str) -> None:
        """Checks that either column format requires both the spread statistic and sample counts."""

        # Remove one required column from an otherwise valid table
        statistics = pd.DataFrame(
            {"nmad": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope")
        ).drop(columns=missing_column)
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # Report the missing column when constructing the magnitude
        with pytest.raises(ValueError, match=f"must contain.*{missing_column}"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

    @pytest.mark.parametrize("named_columns", [False, True])
    def test_variable_from_grouped_stats__error_statistic_requires_explicit_choice(self, named_columns: bool) -> None:
        """Checks that a table with only a custom spread statistic requires an explicit choice."""

        # MAD is a spread estimate, but only std and nmad are recognized automatically
        statistics = pd.DataFrame({"mad": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # Explain how to select the custom spread rather than silently interpreting it
        with pytest.raises(ValueError, match="pass statistic explicitly"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

    @pytest.mark.parametrize("named_columns", [False, True])
    def test_variable_from_grouped_stats__error_missing_requested_statistic(self, named_columns: bool) -> None:
        """Checks that a missing explicit statistic raises an error instead of selecting another estimate."""

        # The requested standard deviation is absent even though nmad could be selected automatically
        statistics = pd.DataFrame({"nmad": [1.0, 3.0], "count": [20, 30]}, index=pd.Index([0.0, 30.0], name="slope"))
        if named_columns:
            statistics.columns = pd.MultiIndex.from_product([["error"], statistics.columns])

        # Require the requested column without falling back to nmad
        with pytest.raises(ValueError, match="must contain.*std"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics, statistic="std")


class TestErrorStructureBindingErrors:
    """Test module for invalid source identities, coordinates, predictors, and covariance indexes."""

    @pytest.mark.parametrize(
        "source_ids, message",
        [
            (np.array([[0, 1]]), "one-dimensional"),
            ([0, 0], "each source_id once"),
            ([[0]], "hashable IDs"),
        ],
    )
    def test_bind__error_invalid_source_ids(self, source_ids: object, message: str) -> None:
        """Checks that bound observations have unique, hashable, one-dimensional source IDs."""

        # Independent errors still need distinct identities to form a covariance matrix
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match=message):
            structure.bind(source_ids)  # type: ignore[arg-type]

    @pytest.mark.parametrize("coordinates", [np.array([0.0, 1.0]), np.array([[0.0], [np.nan]])])
    def test_bind__error_invalid_coordinates(self, coordinates: np.ndarray) -> None:
        """Checks that source coordinates have one finite row per observation."""

        # Coordinate shape and finite values are checked before a spatial model draws errors
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="coordinates must be finite with shape"):
            structure.bind(["a", "b"], coordinates=coordinates)

    def test_bind__error_invalid_predictor(self) -> None:
        """Checks that variable error magnitudes receive one slope per bound observation."""

        # Magnitudes depend on slope at two independent observations
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)])

        # Missing and wrong-length slope values cannot define both magnitudes
        with pytest.raises(ValueError, match="Missing predictors"):
            structure.bind(["a", "b"])
        with pytest.raises(ValueError, match="must be scalar or contain one value per source"):
            structure.bind(["a", "b"], predictors={"slope": [0.0, 0.5, 1.0]})

    def test_covariance_block__error_outside_source(self) -> None:
        """Checks that covariance blocks reject indexes outside the bound observations."""

        # Two source IDs permit positions zero and one only
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        bound = structure.bind(["a", "b"])
        with pytest.raises(ValueError, match="outside the selected source"):
            bound.covariance_block([2], [0])
