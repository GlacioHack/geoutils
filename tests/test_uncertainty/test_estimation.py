"""Tests for estimating error magnitudes and spatial correlations from elevation differences."""

from __future__ import annotations

import warnings
from importlib.util import find_spec

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
