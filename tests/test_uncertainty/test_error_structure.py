"""Tests for error structure classes."""

from __future__ import annotations

import pickle
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from rasterio.transform import from_origin

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.raster.xr_accessor import RasterAccessor
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
        np.testing.assert_allclose(predicted, expected, rtol=1e-14)
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

    def test_predict_magnitude__matches_raster_support(self) -> None:
        """Checks that a predicted magnitude with ``like`` option has shape of a provided raster."""

        # Create synthetic raster and error structure
        values = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
        raster = gu.Raster.from_array(values, (1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # We predict errors for all values using the raster input, should match shape + nodata
        result = structure.predict_magnitude(like=raster)
        assert isinstance(result, gu.Raster)
        np.testing.assert_array_equal(result.to_nanarray(), [[2.0, np.nan], [2.0, 2.0]])
        np.testing.assert_array_equal(raster.to_nanarray(), [[1.0, np.nan], [3.0, 4.0]])

    def test_predict_magnitude__xarray_like(self) -> None:
        """Checks that a magnitude map matches an xarray raster's grid and masked pixels."""

        # A missing source pixel has no predicted error value
        values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        raster = RasterAccessor.from_array(values, transform=from_origin(10, 20, 2, 2), crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Predict on the supplied grid and check coordinates and source values
        result = structure.predict_magnitude(like=raster)
        assert result.shape == raster.shape
        assert result.rst.transform == raster.rst.transform
        assert result.rst.crs == raster.rst.crs
        np.testing.assert_array_equal(result.data, [[2.0, np.nan, 2.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_equal(raster.data, values)

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


class TestErrorStructureChunked:
    """Test module for Dask and multiprocessing error magnitude maps on raster and point support."""

    def test_predict_magnitude__dask_raster_without_predictors(self) -> None:
        """Checks that a constant magnitude leaves an existing Dask raster lazy and preserves its missing pixels."""

        dask = pytest.importorskip("dask.array")

        # The source has shorter final chunks and one pixel without data
        values = np.ones((3, 4))
        values[1, 2] = np.nan
        transform = from_origin(0, 3, 1, 1)
        eager = RasterAccessor.from_array(values, transform=transform, crs=32631)
        lazy = RasterAccessor.from_array(dask.from_array(values, chunks=(2, 3)), transform=transform, crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Predict in the source chunks without calculating the input
        expected = structure.predict_magnitude(like=eager)
        result = structure.predict_magnitude(like=lazy)
        assert hasattr(lazy.data, "compute") and hasattr(result.data, "compute")
        assert result.data.chunks == ((2, 1), (3, 1))

        # Every computed chunk matches the eager map, including the missing pixel
        np.testing.assert_array_equal(np.asarray(result.compute()), np.asarray(expected))

    def test_predict_magnitude__multiband_raster_chunks(self) -> None:
        """Checks that each band has its own predictor values and missing pixel mask across Dask tiles."""

        pytest.importorskip("dask")

        # Two bands have different missing pixels and predictor values on a 3 x 4 grid
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 2, 3] = True
        source = gu.Raster.from_array(
            np.ma.masked_array(np.ones((2, 3, 4)), mask=mask), transform=from_origin(0, 3, 1, 1), crs=32631
        )
        quality = np.stack((np.zeros((3, 4)), np.ones((3, 4))))
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # Predict whole bands and 2 x 3 tiles, leaving shorter final row and column chunks
        expected = structure.predict_magnitude({"quality": quality}, like=source)
        lazy = structure.predict_magnitude({"quality": quality}, like=source, chunksizes=(2, 3))
        assert expected.shape == source.shape == lazy.rst.shape
        assert expected.data.shape == source.data.shape == lazy.data.shape
        assert hasattr(lazy.data, "compute")

        # Each band carries its own magnitude and missing pixel
        reference = np.where(mask, np.nan, np.where(quality == 0, 1.0, 2.0))
        np.testing.assert_array_equal(expected.to_nanarray(), reference)
        np.testing.assert_array_equal(np.asarray(lazy.compute()), reference)

    def test_predict_magnitude__dask_predictors_select_lazy_output(self) -> None:
        """Checks that Dask predictors produce lazy raster and point maps without explicit chunk sizes."""

        dask = pytest.importorskip("dask.array")
        pytest.importorskip("dask_geopandas")

        # The same quality values predict an error from one to two on each spatial input
        quality = np.linspace(0, 1, 7)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        raster = gu.Raster.from_array(np.ones((2, 7)), transform=from_origin(0, 2, 1, 1), crs=32631)
        points = gu.PointCloud.from_xyz(np.arange(7), np.arange(7), np.ones(7), crs=32631)

        # Dask predictor chunks select lazy output even with eager source data
        raster_predictor = dask.from_array(np.tile(quality, (2, 1)), chunks=(1, 3))
        point_predictor = dask.from_array(quality, chunks=3)
        raster_result = structure.predict_magnitude({"quality": raster_predictor}, like=raster)
        point_result = structure.predict_magnitude({"quality": point_predictor}, like=points)
        assert hasattr(raster_result.data, "compute") and not point_result.pc.is_loaded
        assert raster_result.data.chunks == ((1, 1), (3, 3, 1))

        # A scalar predictor also applies to every raster block
        scalar_result = structure.predict_magnitude({"quality": 0.5}, like=raster, chunksizes=(1, 3))
        assert hasattr(scalar_result.data, "compute")
        np.testing.assert_array_equal(np.asarray(scalar_result.compute()), np.full((2, 7), 1.5))

        # Computing both maps matches the same predictions on eager inputs
        eager_raster = structure.predict_magnitude({"quality": np.tile(quality, (2, 1))}, like=raster)
        eager_points = structure.predict_magnitude({"quality": quality}, like=points)
        np.testing.assert_array_equal(np.asarray(raster_result.compute()), eager_raster.to_nanarray())
        np.testing.assert_array_equal(point_result.compute()[points.data_column].to_numpy(), eager_points.data)

    def test_predict_magnitude__raster_dask_mp_equal(self, tmp_path: Path) -> None:
        """Checks that Dask and MP raster tiles predict the eager magnitude map without loading source files."""

        pytest.importorskip("dask")

        # Five rows and seven columns produce shorter edge tiles; a diagonal mask crosses tile boundaries
        values = np.ma.masked_array(np.ones((5, 7)), mask=np.eye(5, 7, dtype=bool))
        source = gu.Raster.from_array(values, transform=from_origin(0, 5, 1, 1), crs=32631, nodata=-9999)
        quality = gu.Raster.from_array(
            np.tile(np.linspace(0, 1, 7), (5, 1)), transform=source.transform, crs=source.crs
        )
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        expected = structure.predict_magnitude({"quality": quality}, like=source)

        # Read raster and predictor tiles from files while keeping both full rasters unloaded
        source_path = tmp_path / "source.tif"
        quality_path = tmp_path / "quality.tif"
        source.to_file(source_path)
        quality.to_file(quality_path)
        file_source = gu.Raster(source_path)
        file_quality = gu.Raster(quality_path)
        lazy = structure.predict_magnitude({"quality": file_quality}, like=file_source, chunksizes=(3, 3))
        assert not file_source.is_loaded and not file_quality.is_loaded
        assert hasattr(lazy.data, "compute")
        assert lazy.data.chunks == ((3, 2), (3, 3, 1))

        # Worker tiles write a file-backed result without loading either complete input
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=(3, 3), outfile=str(tmp_path / "magnitude.tif"), cluster=cluster)
            multiproc = structure.predict_magnitude({"quality": file_quality}, like=file_source, mp_config=config)
        assert not file_source.is_loaded and not file_quality.is_loaded and not multiproc.is_loaded

        # Compare every valid pixel and masked edge with the eager result
        np.testing.assert_array_equal(np.asarray(lazy.compute()), expected.to_nanarray())
        np.testing.assert_array_equal(multiproc.to_nanarray(), expected.to_nanarray())

    def test_predict_magnitude__point_dask_mp_equal(self, tmp_path: Path) -> None:
        """Checks that Dask and MP point partitions predict the eager values in the same row order."""

        dask_geopandas = pytest.importorskip("dask_geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        _register_dask_pointcloud_accessor()

        # Seven ordered points split into 3/3/1 rows so the last partition checks array alignment
        points = gu.PointCloud.from_xyz(np.arange(7), np.arange(7), np.ones(7), crs=32631)
        points.ds["quality"] = np.linspace(0, 1, 7)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        expected = structure.predict_magnitude({"quality": "quality"}, like=points)

        # Existing Dask points and requested row chunks both return unloaded dataframes
        dask_points = dask_geopandas.from_geopandas(points.ds, npartitions=3, sort=False)
        dask_points.pc.data_column = points.data_column
        lazy = structure.predict_magnitude({"quality": "quality"}, like=dask_points)
        requested = structure.predict_magnitude({"quality": "quality"}, like=points, chunksizes=3)
        assert not lazy.pc.is_loaded and not requested.pc.is_loaded
        assert hasattr(lazy, "compute") and hasattr(requested, "compute")

        # Read point rows from a file and write a file-backed result through workers
        source_path = tmp_path / "points.gpkg"
        points.to_file(source_path)
        file_points = gu.PointCloud(source_path, data_column=points.data_column)
        file_lazy = structure.predict_magnitude({"quality": "quality"}, like=file_points, chunksizes=3)
        assert not file_points.is_loaded and not file_lazy.pc.is_loaded
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=3, outfile=str(tmp_path / "magnitude.gpkg"), cluster=cluster)
            multiproc = structure.predict_magnitude({"quality": "quality"}, like=file_points, mp_config=config)
        assert not file_points.is_loaded and not multiproc.is_loaded

        # Compare point values across all partitions and the saved output
        for result in (lazy, requested, file_lazy):
            np.testing.assert_array_equal(result.compute()[points.data_column].to_numpy(), expected.data)
        np.testing.assert_array_equal(multiproc.data, expected.data)

    def test_predict_magnitude__point_geometry_z_chunks(self) -> None:
        """Checks that point magnitudes replace geometry Z when no data column is selected."""

        pytest.importorskip("dask_geopandas")

        # All seven points are 3D, with a shorter final partition of one row
        x = np.arange(7, dtype=float)
        y = np.array([0, 1, 0, 2, 1, 3, 2], dtype=float)
        frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y, np.arange(7)), crs=32631)
        points = gu.PointCloud(frame, data_column=None)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Both backends replace Z with the predicted magnitude without moving points
        expected = structure.predict_magnitude(like=points)
        lazy = structure.predict_magnitude(like=points, chunksizes=3)
        assert not lazy.pc.is_loaded
        result = lazy.compute()
        np.testing.assert_array_equal(result.geometry.x, expected.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, expected.geometry.y)
        np.testing.assert_array_equal(result.geometry.z, expected.geometry.z)


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

    @pytest.mark.parametrize("spatial_type, chunksizes", [("raster", 3), ("point", (2, 2))])
    def test_predict_magnitude__error_invalid_chunks(self, spatial_type: str, chunksizes: Any) -> None:
        """Checks an error is raised for a chunk size that does not match the spatial input."""

        # A raster uses row/column tiles while a point cloud uses row counts
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        if spatial_type == "raster":
            like = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32631)
        else:
            like = gu.PointCloud.from_xyz([0, 1], [0, 1], [1, 1], crs=32631)

        # The wrong chunk option cannot describe that input's output partitions
        with pytest.raises(ValueError, match="chunk size"):
            structure.predict_magnitude(like=like, chunksizes=chunksizes)

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


class TestBoundErrorStructure:
    """Test module for the BoundErrorStructure class."""

    def test_bind__independent_observations(self) -> None:
        """Checks that binding two observations gives their independent covariance matrix."""

        # Two named observations with constant measurement errors
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        bound = structure.bind(["left", "right"])

        # Variance of two squared on the diagonal and zero covariance elsewhere
        assert bound.size == 2
        np.testing.assert_array_equal(bound.source_ids, ["left", "right"])
        np.testing.assert_array_equal(bound.covariance_block([0, 1], [0, 1]), np.diag([4, 4]))

    def test_bind__tuple_source_ids_remain_distinct(self) -> None:
        """Checks that tuple source IDs identify two independent observations in a bound model."""

        # Tuple IDs for two independent observations
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        source_ids = [(1, "a"), (2, "b")]

        # Check diagonal covariance with each tuple treated as one ID
        bound = structure.bind(source_ids)
        covariance = bound.covariance_block([0, 1], [0, 1])
        np.testing.assert_array_equal(covariance, np.diag([4.0, 4.0]))


class TestBoundErrorStructureErrors:
    """Test module for errors/warnings in the BoundErrorStructure class."""

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
    def test_bind__error_invalid_coordinates(self, coordinates: NDArray[np.float64]) -> None:
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
