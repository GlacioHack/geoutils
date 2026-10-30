"""Tests for error structure classes."""

from __future__ import annotations

import pickle
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
import pytest

import geoutils as gu
from geoutils.stats.variography import VariogramModel
from geoutils.uncertainty.error_structure import _source_covariance_matrix


def quadratic_spread(coordinates: Any, offset: float, scale: float) -> Any:
    """Calculate a positive quadratic spread from one or more continuous predictors."""

    return offset + scale * np.sum(np.atleast_2d(coordinates) ** 2, axis=0)


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

    @pytest.mark.parametrize("reverse", [False, True])
    def test_predict__categorical_and_continuous(self, reverse: bool) -> None:
        """Checks that predict() interpolates within categories properly."""

        # We create category codes that are unordered
        angles = pd.IntervalIndex.from_breaks([0, 10, 20], name="angle")
        cover = pd.CategoricalIndex([20, 10, 30], categories=[20, 10, 30], ordered=True, name="cover")
        statistics = pd.DataFrame(
            {"std": [1.0, 10.0, 100.0, 3.0, 30.0, 100.0], "count": [20, 20, 0, 20, 20, 0]},
            index=pd.MultiIndex.from_product([angles, cover]),
        )
        if reverse:
            statistics = statistics.reorder_levels(["cover", "angle"])
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, statistic="std", min_count=10)

        # Check expected output
        # (each class has its separate linear trend, with nearest values beyond the last bins)
        inputs = {"angle": [-5, 10, 25, 10, 10, 10], "cover": [20, 20, 20, 10, 15, 30]}
        expected = [1, 2, 3, 20, np.nan, np.nan]
        np.testing.assert_allclose(magnitude.predict(inputs), expected, equal_nan=True)
        restored = pickle.loads(pickle.dumps(magnitude))
        np.testing.assert_allclose(restored.predict(inputs), expected, equal_nan=True)

    def test_predict__categorical(self) -> None:
        """Checks that predict() selects categorical magnitudes and leaves unknown categories missing."""

        index = pd.CategoricalIndex(["forest", "meadow"], categories=["forest", "meadow"], name="cover")
        statistics = pd.DataFrame({"std": [2.0, 0.5], "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, statistic="std")

        np.testing.assert_allclose(magnitude.predict({"cover": ["meadow", "forest", "shrubs"]}), [0.5, 2, np.nan])
        assert magnitude.predict({"cover": "forest"}) == 2

    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    @pytest.mark.parametrize("model", ["linear", quadratic_spread])
    def test_variable_from_grouped_stats__fitted_model(self, dimensions: int, model: Any) -> None:
        """Checks that affine and custom fits predict known spreads in one or more dimensions."""

        # Create variable errors, with a NaN and a low-count value
        names = [f"predictor_{number}" for number in range(dimensions)]
        index = pd.MultiIndex.from_product([[0.0, 1.0, 2.0, 3.0]] * dimensions, names=names)
        coordinates = np.asarray(index.tolist()).T
        values = 1 + 2 * np.sum(coordinates, axis=0) if model == "linear" else quadratic_spread(coordinates, 1, 2)
        if dimensions == 1:
            index = index.get_level_values(0)
        statistics = pd.DataFrame({"std": values, "count": 20}, index=index)
        statistics.iloc[0, 0] = 1000
        statistics.iloc[0, 1] = 1
        if dimensions > 1:
            statistics.iloc[-1, 0] = np.nan
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(
            statistics, model=model, min_count=10, fit_kwargs={"p0": [1] * (dimensions + 1 if model == "linear" else 2)}
        )

        # We check the fit extrapolates beyond the groups, and works with scalars
        predictors: dict[str, Any] = dict.fromkeys(names, 0.5)
        predictors[names[0]] = np.array([[0.5, 4.0, np.nan]])
        expected = 1 + 2 * (predictors[names[0]] + (dimensions - 1) * 0.5)
        if model != "linear":
            expected = 1 + 2 * (predictors[names[0]] ** 2 + (dimensions - 1) * 0.5**2)
        np.testing.assert_allclose(magnitude.predict(predictors), expected, rtol=1e-7)

    @pytest.mark.parametrize("model", ["piecewise_linear", "linear", quadratic_spread])
    def test_variable_from_grouped_stats__model_options(self, model: Any) -> None:
        """Checks that piecewise linear interpolation is the default, and fitting options work on custom fits."""

        statistics = pd.DataFrame(
            {"std": [1.0, 3.0, 9.0, 19.0], "count": 20}, index=pd.Index([0.0, 1.0, 2.0, 3.0], name="slope")
        )
        options = {"bounds": ([0, 0], [5, 5]), "p0": [1, 1]} if callable(model) else None
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, model=model, fit_kwargs=options)

        if model == "piecewise_linear":
            default = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
            np.testing.assert_array_equal(magnitude.predict({"slope": [0.5, 4]}), default.predict({"slope": [0.5, 4]}))
            np.testing.assert_array_equal(default.predict({"slope": [0.5, 4]}), [2, 19])
        elif model == "linear":
            # Least squares fits a slope of 6 and intercept of -1
            np.testing.assert_allclose(magnitude.predict({"slope": [0, 0.5, 4]}), [np.nan, 2, 23])
        else:
            np.testing.assert_allclose(magnitude.predict({"slope": [0.5, 4]}), [1.5, 33])

    def test_variable_from_grouped_stats__custom_model_by_category(self) -> None:
        """Checks that custom functions fit each category without mixing their spread estimates."""

        cover = pd.CategoricalIndex(["forest", "meadow"], name="cover")
        index = pd.MultiIndex.from_product([cover, [0.0, 1.0, 2.0]], names=["cover", "slope"])
        statistics = pd.DataFrame({"std": [1, 2, 5, 3, 5, 11], "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, model=quadratic_spread)

        np.testing.assert_allclose(
            magnitude.predict({"cover": ["forest", "meadow", "unknown"], "slope": 0.5}), [1.25, 3.5, np.nan]
        )

    @pytest.mark.parametrize("model", ["piecewise_linear", "linear", quadratic_spread])
    def test_pickle__magnitude(self, model: Any) -> None:
        """Checks that an ErrorMagnitude can be pickled."""

        # We create a synthetic error magnitude
        statistics = pd.DataFrame({"std": [1.0, 3.0, 9.0], "count": 20}, index=pd.Index([0.0, 1.0, 2.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, model=model)

        # Try to restore it from pickled
        restored = pickle.loads(pickle.dumps(magnitude))

        # Check equality
        pd.testing.assert_frame_equal(restored.grouped_statistics, magnitude.grouped_statistics)
        predictors = {"slope": [0.0, 0.5, 2.0]}
        np.testing.assert_array_equal(restored.predict(predictors), magnitude.predict(predictors))
        assert restored.model_name == magnitude.model_name


class TestErrorMagnitudeErrors:
    """Test module for errors raised by ErrorMagnitude."""

    @pytest.mark.parametrize(
        "options,message",
        [({"model": "unknown"}, "Magnitude model"), ({"fit_kwargs": {"p0": [1]}}, "fit_kwargs require")],
    )
    def test_variable_from_grouped_stats__error_invalid_model(self, options: dict[str, Any], message: str) -> None:
        """Checks an error is raised for an unknown model or fitting options used with interpolation."""

        statistics = pd.DataFrame({"std": [1, 2, 3], "count": 20}, index=pd.Index([0, 1, 2], name="slope"))

        with pytest.raises(ValueError, match=message):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics, **options)

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
        component = gu.ErrorComponent("uncorrelated", 2)

        # Check argument
        assert component.name == "uncorrelated"
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
            ("uncorrelated", "large", None, TypeError, "must be numeric"),
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

        component = gu.ErrorComponent("uncorrelated", 2)
        structure = gu.ErrorStructure([component])

        # Check attributes
        assert list(structure.components) == ["uncorrelated"]
        assert structure.predict_magnitude() == 2
        assert structure.predict_variance() == 4

    def test_to_covariance_matrix__coincident_independence(self) -> None:
        """Checks that distinct observations at the same coordinates have uncorrelated errors."""

        # We create coincident coordinates
        coordinates = np.array([[1.0, 2.0], [1.0, 2.0]])
        structure = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", 2)])

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
        structure = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", 1)])
        nominal = np.array([10.0, 20.0])

        # Check matching error draws and source values plus errors
        error_samples = list(structure.iter_samples(source_ids=source_ids, n_simulations=2, random_state=4))
        value_samples = list(
            structure.iter_samples(
                source_ids=source_ids, nominal=nominal, kind="value", n_simulations=2, random_state=4
            )
        )
        errors = np.asarray(error_samples)
        values = np.asarray(value_samples)
        assert errors.shape == (2, 2)
        np.testing.assert_array_equal(values, nominal + errors)

    @pytest.mark.parametrize(
        "case, keywords",
        [
            ("constant", ("Constant", "uncorrelated", "gaussian")),
            ("variable", ("Error range", "slope", "piecewise_linear")),
            ("linear", ("Error range", "linear")),
            ("linear_categorical", ("cover", "slope", "linear")),
            ("custom", ("quadratic_spread", "Parameters")),
            ("estimation_missing", ("Magnitude sample size", "nmad", "matheron", "Warning")),
            ("estimation_partial", ("Magnitude sample size", "nmad", "matheron", "Warning")),
            ("estimation_all", ("Magnitude sample size", "nmad", "matheron", "Warning")),
            ("valid_bins", ("Variogram lag bins", "valid")),
            ("sum", ("sum", "matern", "stable")),
            ("product", ("product", "matern", "stable")),
        ],
    )
    @pytest.mark.parametrize("verbose", [False, True])
    def test_info__keywords(
        self, case: str, keywords: tuple[str, ...], verbose: bool, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Checks that info() prints or returns expected component and estimation keywords."""

        # Create the requested error structure, depending on input case
        model = VariogramModel("gaussian", effective_range=100, partial_sill=1)
        components = [gu.ErrorComponent("uncorrelated", 3), gu.ErrorComponent("spatial", 4, model)]
        metadata: dict[str, Any] = {"plot_attrs": {"value": {"units": "m"}}}
        diagnostics: dict[str, Any] = {}
        empirical: gu.Variogram | None = None
        if case == "variable":
            statistics = pd.DataFrame(
                {"std": [3.0, 5.0, np.nan, 100.0], "count": [20, 40, 80, 2]},
                index=pd.Index([0.0, 10.0, 20.0, 30.0], name="slope"),
            )
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(
                statistics, min_count=10, scale=2, variance_offset=36, floor=1
            )
            components = [gu.ErrorComponent("terrain", magnitude)]
        elif case in {"linear", "linear_categorical", "custom"}:
            index = pd.Index([0.0, 1.0, 2.0], name="slope")
            values = np.array([1.0, 3.0, 9.0] if case == "custom" else [2.0, 1.0, 6.0])
            if case == "linear_categorical":
                cover = pd.CategoricalIndex(["forest", "meadow"], name="cover")
                index = pd.MultiIndex.from_product([cover, index])
                values = np.tile(values, 2)
            statistics = pd.DataFrame({"std": values, "count": 20}, index=index)
            magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(
                statistics, model=quadratic_spread if case == "custom" else "linear"
            )
            components = [gu.ErrorComponent("terrain", magnitude)]
        elif case.startswith("estimation_"):
            model = VariogramModel("gaussian", effective_range=30, partial_sill=1)
            components = [gu.ErrorComponent("spatial", 2, model)]
            empirical = gu.Variogram(
                lags=np.array([5.0, 15.0, 25.0]),
                semivariance=np.array([0.1, np.nan, 0.9]),
                counts=np.array([8, 0, 12]),
                bin_lower_edges=np.array([0.0, 10.0, 20.0]),
                bin_edges=np.array([10.0, 20.0, 30.0]),
                estimator="matheron",
                attrs={"distance_units": "m", "n_runs": 2, "sampling": "loglag", "requested_pairs": 999},
            )
            magnitude_diagnostics = {"valid_count": 100}
            sample_count = {"estimation_missing": None, "estimation_partial": 40, "estimation_all": 100}[case]
            if sample_count is not None:
                magnitude_diagnostics["sample_count"] = sample_count
            metadata["spread_estimator"] = "nmad"
            diagnostics = {"magnitude": magnitude_diagnostics, "identifiability": ["Ranges overlap."]}
        elif case == "valid_bins":
            components = [gu.ErrorComponent("noise", 2)]
            empirical = gu.Variogram(
                lags=np.array([5.0, 15.0, 25.0, np.nan]),
                semivariance=np.array([np.nan, 0.1, 0.4, 0.8]),
                counts=np.array([0, 1, 4, 8]),
                estimator="matheron",
            )
        elif case in {"sum", "product"}:
            models = [
                VariogramModel("matern", effective_range=10, partial_sill=1, smoothness=1.5),
                VariogramModel("stable", effective_range=30, partial_sill=3, shape=1.2, active_dims=(0,)),
            ]
            correlation = VariogramModel.combine(models, combination=case)
            components = [gu.ErrorComponent("spatial", 2, correlation)]
        structure = gu.ErrorStructure(
            components, empirical_variogram=empirical, metadata=metadata, fit_diagnostics=diagnostics
        )

        # Check printed and returned summaries
        result = structure.info(verbose=verbose)
        printed = capsys.readouterr().out
        if verbose:
            assert result is None
            summary = printed
        else:
            assert printed == ""
            assert isinstance(result, str)
            summary = result
        for keyword in keywords:
            assert keyword in summary

    @pytest.mark.parametrize("return_axes", [None, False, True])
    def test_plot__return_axes(self, return_axes: bool | None) -> None:
        """Checks that plot() returns the supplied axes only when requested."""

        plt = pytest.importorskip("matplotlib.pyplot")
        structure = gu.ErrorStructure([gu.ErrorComponent("noise", 2)])
        figure, axis = plt.subplots()
        options: dict[str, Any] = {} if return_axes is None else {"return_axes": return_axes}

        result = structure.plot(ax=axis, **options)

        assert [bar.get_height() for bar in axis.patches] == [4]
        assert (result is not None) == bool(return_axes)
        if result is not None:
            assert result["variance"] is axis
        plt.close(figure)

    @pytest.mark.parametrize("magnitude", [0.0, 2.0])
    def test_plot__constant_independent(self, magnitude: float) -> None:
        """Checks that plot() shows the variance of independent errors without empirical diagnostics."""

        plt = pytest.importorskip("matplotlib.pyplot")
        structure = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", magnitude)])

        panels = structure.plot(return_axes=True)

        assert set(panels) == {"variance"}
        assert [bar.get_height() for bar in panels["variance"].patches] == [magnitude**2]
        assert panels["variance"].get_ylabel() == "Error variance"
        plt.close(panels["variance"].figure)

    @pytest.mark.parametrize("kind", ["correlation", "semivariance"])
    @pytest.mark.parametrize("xscale", ["log", "linear"])
    @pytest.mark.parametrize("supplied_axis", [False, True])
    def test_plot__constant_components(self, kind: str, xscale: str, supplied_axis: bool) -> None:
        """Checks that plot() adds component variances and draws the combined correlation model."""

        plt = pytest.importorskip("matplotlib.pyplot")
        pytest.importorskip("skgstat")
        model = VariogramModel("spherical", effective_range=100, partial_sill=1)
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("uncorrelated", 1), gu.ErrorComponent("correlated", 2, model)],
            metadata={"plot_attrs": {"value": {"units": "m"}}},
        )
        frame = plt.subplots()[1] if supplied_axis else None

        panels = structure.plot(ax=frame, kind=kind, xscale=xscale, return_axes=True)

        np.testing.assert_allclose([bar.get_height() for bar in panels["variance"].patches], [1, 4, 5])
        assert panels["variance"].get_ylabel() == "Error variance (m²)"
        axis = panels["correlation"]
        distances = axis.lines[0].get_xdata()
        expected = 4 / 5 * model.correlation(distances)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected if kind == "correlation" else 1 - expected)
        assert axis.get_xscale() == xscale
        assert distances[-1] == 150
        if kind == "correlation":
            assert axis.get_ylim() == (-0.05, 1)
        if xscale == "log":
            assert distances[0] == axis.get_xlim()[0] == 1
        else:
            np.testing.assert_array_equal(axis.collections[0].get_offsets(), [[0, int(kind == "correlation")]])
        if frame is not None:
            assert axis.figure is frame.figure
        plt.close(axis.figure)

    @pytest.mark.parametrize("combined", [False, True])
    @pytest.mark.parametrize("ranges", [(10.0, 100.0), (0.1, 2.0)])
    def test_plot__multiple_model_ranges(self, combined: bool, ranges: tuple[float, float]) -> None:
        """Checks that model plots start at one hundredth of the shortest range, including nested models."""

        plt = pytest.importorskip("matplotlib.pyplot")
        pytest.importorskip("skgstat")
        models = [VariogramModel("gaussian", effective_range=value, partial_sill=1) for value in ranges]
        components = [gu.ErrorComponent(str(index), 1, model) for index, model in enumerate(models)]
        if combined:
            components = [gu.ErrorComponent("spatial", 1, VariogramModel.combine(models, combination="sum"))]

        axis = gu.ErrorStructure(components).plot(show_components=True, return_axes=True)["correlation"]

        assert axis.get_xlim()[0] == pytest.approx(min(ranges) / 100)
        np.testing.assert_allclose(axis.lines[0].get_xdata()[[0, -1]], [min(ranges) / 100, 1.5 * max(ranges)])
        plt.close(axis.figure)

    @pytest.mark.parametrize(
        "show_components,kind,empirical",
        [
            (None, "correlation", False),
            (False, "semivariance", True),
            (True, "correlation", False),
            (True, "semivariance", False),
            (True, "correlation", True),
            (True, "semivariance", True),
        ],
    )
    def test_plot__component_contributions(self, show_components: bool | None, kind: str, empirical: bool) -> None:
        """Checks that optional component curves sum to the combined curve and are hidden by default."""

        plt = pytest.importorskip("matplotlib.pyplot")
        pytest.importorskip("skgstat")
        models = [
            VariogramModel("gaussian", effective_range=10, partial_sill=1),
            VariogramModel("spherical", effective_range=100, partial_sill=4),
        ]
        variogram = None
        if empirical:
            variogram = gu.Variogram(
                lags=np.array([1.0, 10.0]),
                semivariance=np.array([0.2, 0.6]),
                counts=np.array([20, 30]),
                model=VariogramModel.combine(models, combination="sum"),
            )
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("short_range", 1, models[0]), gu.ErrorComponent("long_range", 2, models[1])],
            empirical_variogram=variogram,
        )
        options: dict[str, Any] = {} if show_components is None else {"show_components": show_components}

        axis = structure.plot(kind=kind, return_axes=True, **options)["correlation"]

        curves = [line for line in axis.lines if line.get_label() in {"Short range", "Long range"}]
        assert len(curves) == (2 if show_components else 0)
        if show_components:
            for curve, model, fraction in zip(curves, models, [1 / 5, 4 / 5]):
                expected = fraction * model.correlation(curve.get_xdata())
                if kind == "semivariance":
                    expected = (fraction - expected) * (5 if empirical else 1)
                np.testing.assert_allclose(curve.get_ydata(), expected)
            np.testing.assert_allclose(sum(curve.get_ydata() for curve in curves), axis.lines[-3].get_ydata())
        plt.close(axis.figure)

    @pytest.mark.parametrize(
        "method,kind",
        [("plot", None), ("plot", "semivariance"), ("plot_correlation", None), ("plot_correlation", "correlation")],
    )
    @pytest.mark.parametrize("show_counts", [False, True])
    def test_plot_correlation(self, method: str, kind: str | None, show_counts: bool) -> None:
        """Checks that correlation plots show the requested values and optional counts through either API."""

        plt = pytest.importorskip("matplotlib.pyplot")
        pytest.importorskip("skgstat")
        model = VariogramModel("gaussian", effective_range=3, partial_sill=1)
        empirical = gu.Variogram(
            lags=np.array([1.0, 2.0]),
            semivariance=np.array([0.2, 0.7]),
            counts=np.array([8, 9]),
            model=model,
            estimator="dowd",
        )
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, model)], empirical_variogram=empirical)
        options: dict[str, Any] = {} if kind is None else {"kind": kind}
        if method == "plot":
            options["return_axes"] = True

        result = getattr(structure, method)(show_counts=show_counts, **options)
        axis = result["correlation"] if method == "plot" else result

        quantity = kind or ("correlation" if method == "plot" else "semivariance")
        expected = [0.8, 0.3] if quantity == "correlation" else [0.2, 0.7]
        np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
        assert axis.get_ylabel() == f"Error {quantity}"
        assert "Empirical (Dowd)" in axis.get_legend_handles_labels()[1]
        assert any(item.get_ylabel() == "Pair count" for item in axis.figure.axes) == show_counts
        if quantity == "correlation":
            assert axis.get_ylim() == (-0.05, 1)
        plt.close(axis.figure)

    @pytest.mark.parametrize("named_columns", [False, True])
    @pytest.mark.parametrize("method", ["plot_magnitude", "plot"])
    def test_plot__variable_magnitude(self, named_columns: bool, method: str) -> None:
        """Checks that magnitude plots show bin centers, counts and fitted predictions through either API."""

        plt = pytest.importorskip("matplotlib.pyplot")
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
        options = {"return_axes": True} if method == "plot" else {}
        panels = getattr(structure, method)(**options)
        if method == "plot":
            assert set(panels) == {"magnitude:terrain"}
            panels = panels["magnitude:terrain"]

        # We check the curve uses bin centers, and the count bars show the observations behind each estimate
        empirical, fitted = panels["statistic"].lines
        np.testing.assert_array_equal(empirical.get_xdata(), [5, 15])
        np.testing.assert_array_equal(empirical.get_ydata(), [1, 3])
        np.testing.assert_allclose(fitted.get_ydata(), magnitude.predict({"slope": fitted.get_xdata()}))
        assert [bar.get_height() for bar in panels["count"].patches] == [20, 30]
        assert panels["statistic"].get_xlabel() == "Slope"
        assert panels["statistic"].get_ylabel() == "Error magnitude"
        plt.close(panels["statistic"].figure)

    def test_custom_magnitude__summary_and_plot(self) -> None:
        """Checks that info() and plot_magnitude() describe and evaluate the fitted custom model."""

        plt = pytest.importorskip("matplotlib.pyplot")
        index = pd.IntervalIndex.from_breaks([-0.5, 0.5, 1.5, 2.5], name="slope")
        statistics = pd.DataFrame({"std": [1.0, 3.0, 9.0], "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, model=quadratic_spread)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)])

        summary = structure.info(verbose=False)
        assert summary is not None
        components, estimation = summary.split("Estimation:")
        assert "Functional form: quadratic_spread" in components and "Parameters: 1, 2" in components
        assert "Magnitude bins:" not in components and "Magnitude bins: 3/3 valid" in estimation
        axes = structure.plot_magnitude()
        curve = axes["statistic"].lines[1]
        np.testing.assert_allclose(curve.get_ydata(), quadratic_spread(curve.get_xdata(), 1, 2))
        assert curve.get_label() == "Model (quadratic_spread)"
        plt.close(axes["statistic"].figure)

    @pytest.mark.parametrize("method,show_model", [("plot_magnitude", False), ("plot_magnitude", True), ("plot", True)])
    @pytest.mark.parametrize("plot_kind", ["auto", "bar", "line"])
    def test_plot_magnitude__categorical_bars(self, method: str, show_model: bool, plot_kind: str) -> None:
        """Checks that categorical magnitudes use labelled bars or points and exclude sparse bins."""

        plt = pytest.importorskip("matplotlib.pyplot")
        index = pd.CategoricalIndex([20, 10, 30], categories=[20, 10, 30], ordered=True, name="cover")
        statistics = pd.DataFrame({"std": [0.3, 0.1, 0.2], "count": [120, 50, 140]}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, min_count=100)
        attributes = {"flag_values": [20, 10, 30], "flag_meanings": "forest meadow shrubs"}
        metadata = {"plot_attrs": {"value": {"units": "m"}, "predictors": {"cover": attributes}}}
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)], metadata=metadata)
        options = {"return_axes": True} if method == "plot" else {"show_model": show_model}

        axes = getattr(structure, method)(plot_kind=plot_kind, **options)
        axes = axes["magnitude:terrain"] if method == "plot" else axes
        axis = axes["statistic"]
        if plot_kind == "line":
            values = axis.lines[0].get_ydata()
        else:
            values = [bar.get_height() for bar in axis.containers[0]]
            assert axis.get_ylim()[0] == 0
            assert [text.get_text() for text in axis.texts] == ["0.3", "", "0.2"]
        np.testing.assert_allclose(values, [0.3, np.nan, 0.2])
        assert [bar.get_height() for bar in axes["count"].patches] == [120, 50, 140]
        assert [text.get_text() for text in axis.get_xticklabels()] == ["Forest", "Meadow", "Shrubs"]
        assert axis.get_ylabel() == "Error magnitude (m)"
        assert (axis.get_legend() is not None) == show_model
        plt.close(axis.figure)

    @pytest.mark.parametrize("show_model", [False, True])
    def test_plot_magnitude__continuous_bars(self, show_model: bool) -> None:
        """Checks that an explicit bar plot preserves continuous model predictions."""

        plt = pytest.importorskip("matplotlib.pyplot")
        index = pd.IntervalIndex.from_breaks([0.0, 10.0, 20.0, 30.0], name="angle")
        statistics = pd.DataFrame({"std": [1.0, 3.0, 5.0], "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

        axes = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)]).plot_magnitude(
            plot_kind="bar", show_model=show_model
        )

        axis = axes["statistic"]
        assert [bar.get_height() for bar in axis.containers[0]] == [1, 3, 5]
        assert len(axis.lines) == int(show_model)
        if show_model:
            curve = axis.lines[0]
            np.testing.assert_allclose(curve.get_ydata(), magnitude.predict({"angle": curve.get_xdata()}))
        plt.close(axis.figure)

    @pytest.mark.parametrize("reverse_order", [False, True])
    @pytest.mark.parametrize("show_model", [False, True])
    def test_plot_magnitude__category_curves(self, reverse_order: bool, show_model: bool) -> None:
        """Checks that category curves preserve order, class names, counts and fitted predictions."""

        plt = pytest.importorskip("matplotlib.pyplot")
        angles = pd.IntervalIndex.from_breaks(np.arange(5) - 0.5, name="angle")
        cover = pd.CategoricalIndex([20, 10, 30], categories=[20, 10, 30], ordered=True, name="cover")
        index = pd.MultiIndex.from_product([angles, cover])
        values = (1 + 0.25 * np.asarray(angles.mid)[:, None] ** 2) * np.array([1, 2, 4])
        statistics = pd.DataFrame({"std": values.ravel(), "count": 20}, index=index)
        statistics.iloc[0, statistics.columns.get_loc("count")] = 1
        if reverse_order:
            statistics = statistics.reorder_levels([1, 0])
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics, model=quadratic_spread, min_count=10)
        attributes = {"flag_values": [20, 10, 30], "flag_meanings": "forest meadow shrubs"}
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("terrain", magnitude)], metadata={"plot_attrs": {"predictors": {"cover": attributes}}}
        )

        axes = structure.plot_magnitude(show_model=show_model)

        lines = axes["statistic"].lines
        assert "colorbar" not in axes and len(lines) == 3 * (1 + show_model)
        assert np.isnan(lines[0].get_ydata()[0])
        # Sparse bins stay in counts but are excluded from fitting
        assert [bar.get_width() for bar in axes["count_y"].patches] == [61, 80, 80]
        assert [text.get_text() for text in axes["count_y"].get_yticklabels()] == ["Forest", "Meadow", "Shrubs"]
        assert [text.get_text() for text in axes["count_x"].get_legend().get_texts()] == ["Forest", "Meadow", "Shrubs"]
        if show_model:
            for code, fitted in zip(cover, lines[3:]):
                np.testing.assert_allclose(
                    fitted.get_ydata(), magnitude.predict({"angle": fitted.get_xdata(), "cover": code})
                )
        plt.close(axes["statistic"].figure)

    @pytest.mark.parametrize(
        "method,plot_kind,show_model",
        [
            ("plot_magnitude", "line", False),
            ("plot_magnitude", "line", True),
            ("plot_magnitude", "heatmap", False),
            ("plot_magnitude", "heatmap", True),
            ("plot", "line", True),
            ("plot", "heatmap", True),
        ],
    )
    def test_plot_magnitude__layout_override(self, method: str, plot_kind: str, show_model: bool) -> None:
        """Checks that line and heatmap layouts preserve magnitudes and optional model overlays through either API."""

        plt = pytest.importorskip("matplotlib.pyplot")
        from matplotlib.contour import QuadContourSet

        index = pd.MultiIndex.from_product([[0.0, 1.0], [2.0, 3.0]], names=["angle", "slope"])
        statistics = pd.DataFrame({"std": [1, 2, 3, 4], "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)])
        options = {"return_axes": True} if method == "plot" else {"show_model": show_model}

        axes = getattr(structure, method)(plot_kind=plot_kind, **options)
        axes = axes["magnitude:terrain"] if method == "plot" else axes

        assert ("colorbar" in axes) == (plot_kind == "heatmap")
        if plot_kind == "heatmap":
            np.testing.assert_array_equal(axes["statistic"].collections[0].get_array(), [[1, 3], [2, 4]])
            assert any(isinstance(item, QuadContourSet) for item in axes["statistic"].collections) == show_model
        else:
            lines = axes["statistic"].lines
            assert len(lines) == 2 * (1 + show_model)
            for slope, fitted in zip([2.0, 3.0], lines[2:]):
                angles = np.interp(fitted.get_xdata(), [0.5, 1.5], [0.0, 1.0])
                np.testing.assert_allclose(fitted.get_ydata(), magnitude.predict({"angle": angles, "slope": slope}))
        assert [bar.get_height() for bar in axes["count_x"].patches] == [40, 40]
        plt.close(axes["statistic"].figure)

    @pytest.mark.parametrize("dimensions", [1, 2])
    @pytest.mark.parametrize("categorical", [False, True])
    @pytest.mark.parametrize("empty_labels", [False, True])
    def test_plot_magnitude__labels(self, dimensions: int, categorical: bool, empty_labels: bool) -> None:
        """Checks that supplied axis, count and legend labels override metadata, including empty strings."""

        plt = pytest.importorskip("matplotlib.pyplot")
        first = (
            pd.CategoricalIndex(["low", "high"], name="slope") if categorical else pd.Index([0.0, 10.0], name="slope")
        )
        index = (
            first if dimensions == 1 else pd.MultiIndex.from_product([first, [0.0, 1.0]], names=["slope", "curvature"])
        )
        statistics = pd.DataFrame({"std": np.arange(1, len(index) + 1), "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        metadata = {"plot_attrs": {"value": {"units": "m"}, "predictors": {"slope": {"units": "degrees"}}}}
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude)], metadata=metadata)
        labels: dict[str, Any] = {
            "title": "Binning",
            "xlabel": "Viewing angle (°)",
            "ylabel": "Spread or curvature",
            "cbar_title": "Error (cm)",
            "count_label": "Samples",
        }
        if empty_labels:
            labels = dict.fromkeys(labels, "")

        axes = structure.plot_magnitude(**labels, empirical_label="Binned spread", model_label="Inferred magnitude")

        assert axes["statistic"].get_xlabel() == labels["xlabel"]
        assert axes["statistic"].get_ylabel() == labels["ylabel"]
        counts = axes["count"] if dimensions == 1 else axes["count_x"]
        assert counts.get_title() == labels["title"]
        assert counts.get_ylabel() == labels["count_label"]
        if "colorbar" in axes:
            assert axes["colorbar"].get_xlabel() == labels["cbar_title"]
        if dimensions == 2:
            assert axes["count_y"].get_xlabel() == labels["count_label"]
        expected = ["Binned spread"] + (["Inferred magnitude"] if not categorical or dimensions == 2 else [])
        assert [text.get_text() for text in axes["statistic"].get_legend().get_texts()] == expected
        plt.close(axes["statistic"].figure)

    @pytest.mark.parametrize("dimensions", [1, 2])
    @pytest.mark.parametrize("supplied_axis", [False, True])
    def test_plot__combined_diagnostics(self, dimensions: int, supplied_axis: bool) -> None:
        """Checks that magnitude and correlation diagnostics share a figure and appear side by side."""

        plt = pytest.importorskip("matplotlib.pyplot")
        pytest.importorskip("skgstat")
        intervals = pd.IntervalIndex.from_breaks([0.0, 10.0, 20.0], name="slope")
        index = (
            intervals
            if dimensions == 1
            else pd.MultiIndex.from_product([intervals, [0.0, 1.0]], names=["slope", "curvature"])
        )
        statistics = pd.DataFrame({"nmad": np.arange(1, len(index) + 1), "count": 20}, index=index)
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        model = VariogramModel("gaussian", effective_range=3, partial_sill=1)
        empirical = gu.Variogram(
            lags=np.array([1.0, 2.0]), semivariance=np.array([0.2, 0.7]), counts=np.array([8, 9]), model=model
        )
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude, model)], empirical_variogram=empirical)
        frame = plt.subplots(figsize=(12, 4.5))[1] if supplied_axis else None

        panels = structure.plot(ax=frame, return_axes=True)
        magnitude_axes, correlation_axis = panels["magnitude:terrain"], panels["correlation"]
        figure = correlation_axis.figure
        figure.canvas.draw()

        assert set(panels) == {"magnitude:terrain", "correlation"}
        assert all(axis.figure is figure for axis in magnitude_axes.values())
        assert magnitude_axes["statistic"].get_position().x1 < correlation_axis.get_position().x0
        assert magnitude_axes["statistic"].get_position().y0 == pytest.approx(correlation_axis.get_position().y0)
        assert correlation_axis.get_ylabel() == "Error correlation"
        if frame is not None:
            assert figure is frame.figure
        plt.close(figure)


class TestSourceCovariance:
    """Test module for covariance between observations."""

    def test_source_covariance_matrix__rectangular_observation_groups(self) -> None:
        """Checks that covariance handles repeated observations."""

        pytest.importorskip("skgstat")

        # Create observations at two locations
        source_ids = np.array([10, 11, 12])
        coordinates = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
        correlation = VariogramModel("spherical", effective_range=10, partial_sill=1)
        independent = gu.ErrorComponent("uncorrelated", 2.0)
        spatial = gu.ErrorComponent("spatial", 1.0, correlation)
        component_data = ((independent, np.full(3, 2.0)), (spatial, np.ones(3)))

        # Calculate covariance for reordered and repeated observations
        rows = np.array([0, 2])
        columns = np.array([1, 0, 2, 0])
        covariance = _source_covariance_matrix(source_ids, coordinates, component_data, rows, columns)

        # Check covariance includes independent errors only for the same observation
        rho = float(spatial.predict_correlation(5.0))
        expected = np.array([[1.0, 5.0, rho, 5.0], [rho, rho, 5.0, rho]])
        np.testing.assert_allclose(covariance, expected)


class TestErrorStructureErrors:
    """Test module for errors/warnings of the ErrorStructure class."""

    def test_plot_correlation__error_missing_empirical_variogram(self) -> None:
        """Checks an error is raised for plot_correlation() without an empirical variogram."""

        structure = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", 1)])

        with pytest.raises(ValueError, match="No empirical variogram"):
            structure.plot_correlation()

    @pytest.mark.parametrize("options", [{"kind": "unknown"}, {"xscale": "unknown"}])
    def test_plot__error_invalid_model_options(self, options: dict[str, Any]) -> None:
        """Checks an error is raised for an invalid plot quantity or distance scale."""

        plt = pytest.importorskip("matplotlib.pyplot")

        model = VariogramModel("spherical", effective_range=100, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, model)])
        figure, frame = plt.subplots()

        try:
            with pytest.raises(ValueError, match="Argument"):
                structure.plot(ax=frame, **options)
        finally:
            plt.close(figure)

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"source_ids": ["a", "b"], "n_simulations": 0}, "n_simulations must be a positive integer"),
            ({"source_ids": ["a", "b"], "kind": "unknown"}, "kind must"),
            ({"source_ids": ["a", "b"], "nominal": [1.0]}, "nominal must contain one value"),
        ],
    )
    def test_iter_samples__error_invalid_request(self, options: dict[str, object], message: str) -> None:
        """Checks an error is raised for an invalid sample count, kind, or source values."""

        structure = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", 1)])
        with pytest.raises(ValueError, match=message):
            list(structure.iter_samples(**options))  # type: ignore[arg-type]
