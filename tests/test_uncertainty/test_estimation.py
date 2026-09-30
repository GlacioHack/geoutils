"""Tests for estimating error magnitudes and spatial correlations from elevation differences."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter

import geoutils as gu
from geoutils.stats.variography import VariogramModel


class TestErrorStructureEstimation:
    """Test module for estimating independent and correlated error components."""

    def test_estimate__constant_independent_component(self) -> None:
        """Checks that independent errors have the standard deviation of the centered proxy values."""

        # Symmetric point errors have zero median and standard deviation sqrt(2.5)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Fit one constant component without a spatial variogram
        structure = gu.ErrorStructure.estimate(
            proxy,
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
            random_state=2,
        )

        # Population variance is the mean of 4, 1, 1, and 4
        assert structure.predict_magnitude() == pytest.approx(np.sqrt(np.mean(values**2)))
        assert structure.empirical_variogram is None
        assert structure.fit_diagnostics["refinement"]["success"] is None

    def test_estimate_independent_variable_component_without_variography(self) -> None:
        """Checks that an independent component recovers increasing magnitudes without fitting a variogram."""

        # Errors with spread increasing with quality
        rng = np.random.default_rng(5)
        predictor = np.broadcast_to(np.linspace(0, 1, 40), (40, 40))
        values = 20 + (0.5 + predictor) * rng.normal(size=predictor.shape)

        # Proxy and predictor on same grid
        transform = Affine(10, 0, 0, 0, -10, 400)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor_raster = gu.Raster.from_array(predictor, transform, 32632, nodata=-9999)

        # Fit variable magnitude (independent errors)
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"quality": predictor_raster},
            components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            bins=5,
            min_count=50,
            random_state=4,
        )

        # Check increasing error spread with quality
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert predicted[1] > predicted[0]

        # Check empty variogram/refinement diagnostics for independent errors
        assert structure.empirical_variogram is None
        assert structure.fit_diagnostics["refinement"]["success"] is None

    def test_estimate_point_predictor_on_masked_common_support(self) -> None:
        """Checks that point predictor columns and a Boolean mask use the same finite observations."""

        # Point errors with spread set by quality column
        rng = np.random.default_rng(11)
        positions = np.arange(240, dtype=float)
        quality = np.linspace(0, 1, len(positions))
        values = (0.5 + quality) * rng.normal(size=len(positions))
        proxy = gu.PointCloud.from_xyz(positions, np.zeros_like(positions), values, crs=32632)
        proxy.ds["quality"] = quality

        # Exclude masked points and missing predictor together
        selected = np.ones(len(positions), dtype=bool)
        selected[:40] = False
        proxy.ds.loc[100, "quality"] = np.nan
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"quality": "quality"},
            components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            mask=selected,
            bins=4,
            min_count=30,
            random_state=5,
        )

        # Check increasing fitted spread on selected rows
        predicted = structure.predict_magnitude({"quality": np.array([0.25, 0.9])})
        assert predicted[1] > predicted[0]

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_estimate_separates_variable_short_and_fixed_long_components(self) -> None:
        """Checks that estimation separates a variable short range error from a constant long range error."""

        # Random fields with short/long smoothing lengths
        rng = np.random.default_rng(9)
        predictor = np.broadcast_to(np.linspace(0, 1, 48), (48, 48))
        short_field = gaussian_filter(rng.normal(size=predictor.shape), 1)
        long_field = gaussian_filter(rng.normal(size=predictor.shape), 7)

        # Normalize fields before scaling error magnitudes
        short_field = (short_field - short_field.mean()) / short_field.std()
        long_field = (long_field - long_field.mean()) / long_field.std()
        values = (0.5 + 1.5 * predictor) * short_field + 0.7 * long_field

        # Combined errors and quality predictor on same grid
        transform = Affine(10, 0, 0, 0, -10, 480)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor_raster = gu.Raster.from_array(predictor, transform, 32632, nodata=-9999)

        # Fit default two-component model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": predictor_raster},
                bins=4,
                min_count=100,
                n_pairs=5_000,
                n_lags=8,
                random_state=2,
            )

        # Check variable short-range and constant long-range models
        short = structure.components["short_range"]
        long = structure.components["long_range"]
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert isinstance(short.magnitude, gu.ErrorMagnitude)
        assert isinstance(long.magnitude, gu.ErrorMagnitude)
        assert isinstance(short.correlation, VariogramModel)
        assert isinstance(long.correlation, VariogramModel)
        assert short.correlation.effective_range is not None and long.correlation.effective_range is not None
        assert short.magnitude.kind == "variable"
        assert long.magnitude.kind == "constant"

        # Check range order, increasing spread and successful refinement
        assert short.correlation.effective_range < long.correlation.effective_range
        assert predicted[1] > predicted[0]
        assert structure.fit_diagnostics["refinement"]["success"]

    def test_variogram_estimation_retains_only_compact_pair_diagnostics(self) -> None:
        """Checks that estimation retains grouped diagnostics and can refit a model without storing sampled pairs."""

        # Require optional variogram fitting backend
        pytest.importorskip("skgstat")

        # Smooth grid variation at two spatial scales
        y, x = np.mgrid[:40, :40]
        values = np.sin(x / 4) + 0.5 * np.cos(y / 10)
        proxy = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 400), 32632, nodata=-9999)

        # Fit Gaussian correlation, then refit bins with spherical model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                proxy,
                components={"spatial": {"magnitude": "constant", "correlation": "gaussian"}},
                n_pairs=1_000,
                n_lags=6,
                refine=True,
                random_state=42,
            )
            refitted = structure.refit("spherical")

        # Check diagnostics per lag bin
        refinement = structure.fit_diagnostics["refinement"]
        conditional = refinement["conditional_statistics"]
        assert refinement["success"]
        assert isinstance(conditional, pd.DataFrame)
        assert set(conditional) == {"lag", "mean_magnitude", "semivariance", "fitted_semivariance", "count"}
        assert len(conditional) <= 6

        # Check discarded pair arrays and fitted spherical model
        assert not any("pair" in name for name in vars(structure))
        refitted_model = refitted.components["spatial"].correlation
        assert isinstance(refitted_model, VariogramModel)
        assert refitted_model.model_name == "spherical"


class TestStandardization:
    """Test module for calibrating magnitudes and rejecting standardized outliers."""

    def test_two_step_standardization(self) -> None:
        """Checks that two-step standardization removes an outlier and rescales the remaining errors to unit spread."""

        from geoutils.uncertainty.estimation import two_step_standardization

        # Variable error spread with extreme outlier
        rng = np.random.default_rng(1)
        quality = rng.uniform(0, 1, 1000)
        values = rng.normal(size=1000) * (1 + quality)
        values[0] = 1000

        # Double magnitude model to check scale correction
        def unscaled(predictors: tuple[ArrayLike, ...]) -> NDArray[np.float64]:
            """Return an intentionally oversized magnitude for the quality predictor."""

            return 2.0 * (1.0 + np.asarray(predictors[0], dtype=float))

        # Standardize errors and get corrected magnitude model
        standardized, model = two_step_standardization(values, [quality], unscaled)

        # Check outlier rejection, unit spread and recovered errors
        assert np.isnan(standardized[0])
        assert gu.stats.nmad(standardized) == pytest.approx(1)
        assert model((quality,))[1:] == pytest.approx(values[1:] / standardized[1:])

    def test_two_step_standardization__masked_outlier(self) -> None:
        """Checks that a masked proxy stays masked and an extreme standardized error is excluded."""

        from geoutils.uncertainty.estimation import two_step_standardization

        # Four ordinary errors, one extreme error, and one missing observation
        values = np.ma.array([-2.0, -1.0, 1.0, 2.0, 100.0, 0.0], mask=[False] * 5 + [True])

        def unit_magnitude(predictors: tuple[ArrayLike, ...]) -> NDArray[np.float64]:
            """Return unit error magnitude at every supplied observation."""

            return np.ones_like(np.asarray(predictors[0], dtype=float))

        # Standardize after masking the extreme value
        standardized, _ = two_step_standardization(values, [np.arange(len(values), dtype=float)], unit_magnitude)

        # Four ordinary errors set the spread; the extreme and missing values remain masked
        np.testing.assert_array_equal(np.ma.getmaskarray(standardized), [False] * 4 + [True, True])
        assert gu.stats.nmad(standardized) == pytest.approx(1)


class TestErrorStructureEstimationErrors:
    """Test module for invalid configurations and statistical controls in ErrorStructure.estimate()."""

    @pytest.mark.parametrize(
        "components, error_type, message",
        [
            ({}, ValueError, "at least one named"),
            ({"": {}}, TypeError, "non-empty names"),
            ({"measurement": {"unknown": 1}}, ValueError, "Unknown configuration"),
            ({"measurement": {"magnitude": "invalid"}}, ValueError, "magnitude must"),
            ({"measurement": {"magnitude": "heteroscedastic"}}, ValueError, "requires at least one"),
            ({"measurement": {"correlation": 2}}, TypeError, "variogram model name"),
        ],
    )
    def test_estimate__error_invalid_component_configuration(
        self, components: dict[str, dict[str, Any]], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for a component description that cannot define an error model."""

        # A valid point proxy isolates validation of the component description
        proxy = gu.PointCloud.from_xyz([0, 1, 2, 3], [0, 0, 0, 0], [-2, -1, 1, 2], crs=32631)

        # Invalid names, magnitudes, and correlation types fail before fitting
        with pytest.raises(error_type, match=message):
            gu.ErrorStructure.estimate(proxy, components=components)

    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"fit_method": "unknown"}, NotImplementedError, "Only fit_method"),
            ({"min_count": 0}, ValueError, "min_count must"),
            ({"outlier_factor": 0}, ValueError, "outlier_factor must"),
            ({"spread_estimator": np.min}, ValueError, "spread_estimator returned"),
            (
                {"components": {"first": {"correlation": None}, "second": {"correlation": None}}},
                ValueError,
                "Only one independent component",
            ),
        ],
    )
    def test_estimate__error_invalid_fitting_options(
        self, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid fitting controls or multiple independent components."""

        # Four finite point errors make a valid constant-spread baseline
        proxy = gu.PointCloud.from_xyz([0, 1, 2, 3], [0, 0, 0, 0], [-2, -1, 1, 2], crs=32631)
        default: dict[str, Any] = {"components": {"measurement": {"correlation": None}}, "spread_estimator": np.std}

        # Change one option at a time and check its reported error
        with pytest.raises(error_type, match=message):
            gu.ErrorStructure.estimate(proxy, **(default | options))
