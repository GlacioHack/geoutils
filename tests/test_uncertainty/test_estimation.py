"""Tests for estimating error magnitudes and spatial correlations from elevation differences."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from scipy.ndimage import gaussian_filter

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.stats.variography import VariogramModel


def quadratic_magnitude(values: Any, offset: float, coefficient: float) -> Any:
    """Calculate an error magnitude that increases quadratically with its predictor."""

    return offset + coefficient * np.asarray(values) ** 2


def exponential_variogram(distance: Any, effective_range: float, partial_sill: float) -> Any:
    """Calculate an exponential semivariance reaching about 95% of its sill at the effective range."""

    return partial_sill * (1 - np.exp(-3 * np.asarray(distance) / effective_range))


@pytest.fixture
def pointcloud_file(tmp_path: Any) -> Any:
    """Write point fixtures as GeoParquet row groups with their original labels and attribute dtypes."""

    import geopandas as gpd

    pytest.importorskip("pyarrow")
    filenames = []

    def write_points(frame: gpd.GeoDataFrame, partitions: int) -> tuple[str, int, str | None]:
        """Save one point fixture and return its filename, row chunk size and active value column."""

        # Preserve explicit elevation selection, otherwise infer the common point value columns
        frame = frame.copy()
        column = frame.pc.data_name
        if "data_name" not in frame.attrs and column is None:
            column = next((name for name in ("height", "z", "intensity") if name in frame), None)
        if column is None and not frame.geometry.has_z.all():
            frame["z"] = np.zeros(len(frame))
            column = "z"

        # A final shorter row group checks that file readers preserve every row
        chunks = max(1, len(frame) // partitions + 1)
        filename = tmp_path / f"points-{len(filenames)}.parquet"
        filenames.append(filename)
        frame.to_parquet(
            filename, index=True, geometry_encoding="geoarrow", schema_version="1.1.0", row_group_size=chunks
        )
        return str(filename), chunks, column

    return write_points


class TestErrorStructureEstimation:
    """Test module for estimating independent and correlated error components."""

    def test_estimate__constant_independent_component(self) -> None:
        """Checks that independent errors have the standard deviation of the proxy values."""

        # Symmetric point errors have zero median and standard deviation sqrt(2.5)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Fit one constant component without a spatial variogram
        structure = gu.ErrorStructure.estimate(
            source=proxy,
            components={"uncorrelated": {"magnitude": "constant", "correlation": None}},
            random_state=2,
        )

        # Population variance is the mean of 4, 1, 1, and 4
        assert structure.predict_magnitude() == pytest.approx(np.sqrt(np.mean(values**2)))
        assert structure.empirical_variogram is None

    @pytest.mark.parametrize(
        "representation", ["raster", "raster_accessor", "pointcloud", "geometry_accessor", "array_accessor"]
    )
    def test_estimate_error_structure__proxy(self, representation: str) -> None:
        """Checks that dataset methods estimate the same error proxy as ErrorStructure.estimate() without other."""

        values = np.array([-2.0, -1.0, 1.0, 2.0])
        raster = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631)
        points = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)
        interfaces = {
            "raster": raster,
            "raster_accessor": raster.to_xarray().rst,
            "pointcloud": points,
            "geometry_accessor": points.gdf.pc,
            "array_accessor": points.to_xarray().pc,
        }
        source = interfaces[representation]
        components = {"uncorrelated": {"magnitude": "constant", "correlation": None}}

        result = source.estimate_error_structure(components=components)
        expected = gu.ErrorStructure.estimate(source, components=components)

        assert result.predict_magnitude() == expected.predict_magnitude() == np.std(values)
        assert result.empirical_variogram is None

    @pytest.mark.parametrize("variable", [False, True])
    @pytest.mark.parametrize("subsample, expected_count", [(1, 20), (8, 8), (0.5, 10)])
    def test_estimate__magnitude_sample_size(self, variable: bool, subsample: int | float, expected_count: int) -> None:
        """Checks that spread diagnostics count the observations used after subsampling."""

        # Twenty finite observations provide samples with known sizes
        values = np.tile([-2.0, -1.0, 1.0, 2.0], 5)
        proxy = gu.PointCloud.from_xyz(np.arange(20), np.zeros(20), values, crs=32631)
        predictors = {"quality": np.linspace(0, 1, 20)} if variable else None
        components = {"uncorrelated": {"magnitude": "heteroscedastic" if variable else "constant", "correlation": None}}

        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors=predictors,
            components=components,
            bins=2,
            min_count=2,
            subsample_magnitude=subsample,
            random_state=42,
        )

        assert structure.fit_diagnostics["magnitude"] == {"valid_count": 20, "sample_count": expected_count}
        summary = structure.info(verbose=False)
        assert summary is not None
        expected_sample = f"{expected_count} of 20" if expected_count < 20 else "20"
        assert "Magnitude estimator: std" in summary
        assert f"Magnitude sample size: {expected_sample}" in summary
        assert "Variogram estimator: Not estimated" in summary

    def test_estimate__magnitude_sample_size_with_excluded_bins(self) -> None:
        """Checks that spread sample size excludes observations outside the supplied predictor bins."""

        # Predictor bins cover the first half of the twenty observations
        values = np.tile([-2.0, -1.0, 1.0, 2.0], 5)
        proxy = gu.PointCloud.from_xyz(np.arange(20), np.zeros(20), values, crs=32631)
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"quality": np.linspace(0, 1, 20)},
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            bins={"quality": [0, 0.25, 0.5]},
            min_count=2,
        )

        assert structure.fit_diagnostics["magnitude"] == {"valid_count": 20, "sample_count": 10}

    def test_estimate__robust_spread(self) -> None:
        """Checks that an explicit NMAD estimator limits the influence of an outlier."""

        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 100.0])
        proxy = gu.PointCloud.from_xyz(np.arange(len(values)), np.zeros(len(values)), values, crs=32631)
        structure = gu.ErrorStructure.estimate(
            proxy,
            components={"uncorrelated": {"correlation": None}},
            spread_estimator=gu.stats.nmad,
        )

        assert structure.predict_magnitude() == pytest.approx(gu.stats.nmad(values))
        assert structure.predict_magnitude() < np.std(values)

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("other_precision", [None, "same", "negligible"])
    @pytest.mark.parametrize("entrypoint", ["class", "method"])
    def test_estimate_error_structure__two_inputs(
        self, kind: str, other_precision: str | None, entrypoint: str
    ) -> None:
        """Checks that two aligned inputs yield the expected error magnitude for either precision assumption."""

        # Measurements differ by known errors; one reference value is missing
        errors = np.array([-2.0, -1.0, 1.0, 2.0])
        reference = np.array([10.0, 10.0, 10.0, np.nan])
        measured = np.full(4, 10.0) + errors
        if kind == "raster":
            transform = Affine(10, 0, 0, 0, -10, 20)
            source = gu.Raster.from_array(measured.reshape(2, 2), transform, 32631, nodata=-9999)
            other = gu.Raster.from_array(reference.reshape(2, 2), transform, 32631, nodata=-9999)
        else:
            x = np.arange(4, dtype=float)
            source = gu.PointCloud.from_xyz(x, np.zeros(4), measured, crs=32631)
            other = gu.PointCloud.from_xyz(x, np.zeros(4), reference, crs=32631)

        # The shared finite support has three errors; equal precision splits their variance in half
        options: dict[str, Any] = {
            "other": other,
            "components": {"uncorrelated": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
        }
        if other_precision is not None:
            options["other_precision"] = other_precision

        # Estimate through the class with input keywords or the dataset method
        if entrypoint == "class":
            structure = gu.ErrorStructure.estimate(source=source, **options)
        else:
            structure = source.estimate_error_structure(**options)

        # Check the magnitude and that no spatial correlation was fitted
        scale = np.sqrt(2) if other_precision == "same" else 1.0
        assert structure.predict_magnitude() == pytest.approx(np.std(errors[:3]) / scale)
        assert structure.empirical_variogram is None

    @pytest.mark.parametrize("at", ["other", None])
    def test_estimate__raster_errors_at_comparison_points(self, at: Literal["other"] | None) -> None:
        """Checks that raster errors can be estimated at comparison points."""

        # Create a raster and comparison points with one NaN
        values = np.arange(36, dtype=float).reshape(6, 6)
        source = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999)
        rows, columns = np.array([1, 1, 4, 4]), np.array([1, 4, 1, 4])
        x, y = source.ij2xy(rows, columns)
        reference = gu.PointCloud.from_xyz(x, y, np.array([0.0, 0.0, np.nan, 0.0]), crs=32632)

        # Estimate errors at comparison points
        structure = gu.ErrorStructure.estimate(
            source,
            other=reference,
            other_precision="negligible",
            at=at,
            components={"uncorrelated": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
        )

        # Check the standard deviation of the three finite comparisons
        expected_errors = values[rows, columns][[0, 1, 3]]
        assert structure.predict_magnitude() == pytest.approx(np.std(expected_errors))

    def test_estimate__reprojected_comparison_grid(self) -> None:
        """Checks that estimation can reproject a comparison raster."""

        # Create rasters with different resolutions and a zero reference
        values = np.arange(36, dtype=float).reshape(6, 6)
        source = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 60), 32632, nodata=-9999)
        reference = gu.Raster.from_array(np.zeros((3, 3)), Affine(20, 0, 0, 0, -20, 60), 32632, nodata=-9999)

        # Estimate errors after reprojection
        structure = gu.ErrorStructure.estimate(
            source,
            other=reference,
            other_precision="negligible",
            align="reproject",
            components={"uncorrelated": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
        )

        # Check equality with the source standard deviation
        assert structure.predict_magnitude() == pytest.approx(np.std(values))

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
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            bins=5,
            min_count=50,
            random_state=4,
        )

        # Check increasing error spread with quality
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert predicted[1] > predicted[0]

        # Check no variogram is fitted for independent errors
        assert structure.empirical_variogram is None

    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    @pytest.mark.parametrize("custom", [False, True])
    def test_estimate__magnitude_model(self, dimensions: int, custom: bool) -> None:
        """Checks that estimate() fits an affine or custom magnitude from grouped predictor spreads."""

        # Balanced errors give each group an exact population standard deviation
        coordinates = np.asarray(list(np.ndindex(*(3,) * dimensions)), dtype=float).T
        spread = 1 + np.sum(coordinates**2 if custom else coordinates, axis=0)
        values = np.column_stack([-spread, spread]).ravel()
        proxy = gu.PointCloud.from_xyz(np.arange(len(values)), np.zeros(len(values)), values, crs=32631)
        predictors = {f"predictor_{number}": np.repeat(coordinate, 2) for number, coordinate in enumerate(coordinates)}

        def model(x: Any, offset: float, scale: float) -> Any:
            """Calculate a quadratic magnitude across all continuous predictors."""

            return offset + scale * np.sum(np.atleast_2d(x) ** 2, axis=0)

        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors=predictors,
            bins=dict.fromkeys(predictors, [-0.5, 0.5, 1.5, 2.5]),
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            min_count=2,
            magnitude_model=model if custom else "linear",
            magnitude_fit_kwargs={"p0": [1] * (2 if custom else dimensions + 1)},
        )

        np.testing.assert_allclose(structure.predict_magnitude(predictors), np.repeat(spread, 2), rtol=1e-7)
        magnitude = structure.components["uncorrelated"].magnitude
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert magnitude.model_name == ("model" if custom else "linear")
        assert structure.fit_diagnostics["magnitude"]["sample_count"] == len(values)

    @pytest.mark.parametrize("object_type", ["raster", "dataarray"])
    @pytest.mark.parametrize("unit", ["m", "cm"])
    def test_estimate__plot_labels(self, object_type: str, unit: str) -> None:
        """Checks that plot() labels estimated magnitudes and predictors using input names and units."""

        pyplot = pytest.importorskip("matplotlib.pyplot")

        # Create errors with varying magnitude
        angles = np.broadcast_to(np.linspace(0, 60, 8), (8, 8)).copy()
        values = (1 + angles / 60) * np.tile([-1.0, 1.0], (8, 4))
        transform = Affine(10, 0, 0, 0, -10, 80)
        constructor = gu.Raster.from_array if object_type == "raster" else gu.raster.from_array
        proxy = constructor(values, transform, 32631)
        predictor = constructor(angles, transform, 32631)
        proxy_interface = proxy if object_type == "raster" else proxy.rst
        predictor_interface = predictor if object_type == "raster" else predictor.rst
        proxy_interface.tags.update(long_name="Snow depth", units=unit)
        predictor_interface.tags.update(long_name="Viewing angle", units="degrees")

        # Estimate and plot without specifying labels
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"view_angle": predictor},
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            bins=4,
            min_count=8,
            spread_estimator=np.std,
            random_state=42,
        )
        axes = structure.plot(return_axes=True)["magnitude:uncorrelated"]

        assert axes["statistic"].get_xlabel() == "Viewing angle (°)"
        assert axes["statistic"].get_ylabel() == f"Error magnitude ({unit})"
        assert axes["count"].get_title() == "Error variability"
        pyplot.close(axes["statistic"].figure)

    def test_estimate__categorical_and_continuous_predictors(self) -> None:
        """Checks that estimation fits separate viewing-angle trends for categorical land cover."""

        pyplot = pytest.importorskip("matplotlib.pyplot")

        # Each category has the same angles and balanced positive and negative errors
        angles = np.broadcast_to(np.linspace(0, 60, 18), (18, 18)).copy()
        codes = np.repeat([20, 10, 30], 6)
        cover = np.broadcast_to(codes[:, None], angles.shape).copy()
        scales = np.repeat([1.0, 2.0, 4.0], 6)[:, None]
        signs = np.tile([-1.0, 1.0], 9)[:, None]
        values = scales * (1 + angles / 60) * signs
        transform = Affine(10, 0, 0, 0, -10, 180)
        proxy = gu.raster.from_array(values, transform, 32631).assign_attrs(units="m")
        angle_raster = gu.raster.from_array(angles, transform, 32631)
        cover_raster = gu.raster.from_array(cover, transform, 32631).assign_attrs(
            long_name="Land cover", flag_values=[20, 10, 30], flag_meanings="forest meadow shrubs"
        )

        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"view_angle": angle_raster, "land_cover": cover_raster},
            bins={"view_angle": 3},
            categories={"land_cover": [20, 10, 30]},
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            spread_estimator=np.std,
            min_count=10,
            random_state=42,
        )

        # The class scales multiply the same viewing-angle trend
        predicted = structure.predict_magnitude({"view_angle": 30, "land_cover": [20, 10, 30]})
        np.testing.assert_allclose(predicted / predicted[0], [1, 2, 4])
        axes = structure.plot(return_axes=True)["magnitude:uncorrelated"]
        assert [label.get_text() for label in axes["count_y"].get_yticklabels()] == ["Forest", "Meadow", "Shrubs"]
        assert axes["statistic"].get_ylabel() == "Error magnitude (m)"
        pyplot.close(axes["statistic"].figure)

    def test_estimate_point_predictor_on_masked_common_support(self) -> None:
        """Checks that point predictor columns and a Boolean mask use the same finite observations."""

        # Point errors with spread set by quality column
        rng = np.random.default_rng(11)
        positions = np.arange(240, dtype=float)
        quality = np.linspace(0, 1, len(positions))
        values = (0.5 + quality) * rng.normal(size=len(positions))
        proxy = gu.PointCloud.from_xyz(positions, np.zeros_like(positions), values, crs=32632)
        proxy.gdf["quality"] = quality

        # Exclude masked points and missing predictor together
        selected = np.ones(len(positions), dtype=bool)
        selected[:40] = False
        proxy.gdf.loc[100, "quality"] = np.nan
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"quality": "quality"},
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            mask=selected,
            bins=4,
            min_count=30,
            random_state=5,
        )

        # Check increasing fitted spread on selected rows
        predicted = structure.predict_magnitude({"quality": np.array([0.25, 0.9])})
        assert predicted[1] > predicted[0]

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    @pytest.mark.parametrize("min_count", [None, 1, 200])
    def test_estimate__minimum_pairs_per_bin(self, min_count: int | None) -> None:
        """Checks that estimation defaults to 100 pairs per bin and refit() preserves or overrides that threshold."""

        pytest.importorskip("skgstat")
        rows, columns = np.indices((32, 32))
        values = np.sin(columns / 4) + 0.5 * np.cos(rows / 8)
        proxy = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 320), 32631)
        options = {} if min_count is None else {"min_count": min_count}
        minimum = 100 if min_count is None else min_count

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                proxy,
                components={"spatial": {"magnitude": "constant", "correlation": "gaussian"}},
                n_pairs=5_000,
                n_lags=10,
                pair_sampling="random_xy",
                max_lag=200,
                random_state=42,
                fit_kwargs=options,
            )
            refitted = structure.refit("spherical")
            overridden = structure.refit("spherical", fit_kwargs={"min_count": 1})

        empirical = structure.empirical_variogram
        assert empirical is not None
        valid = (empirical.counts >= minimum) & np.isfinite(empirical.lags) & np.isfinite(empirical.semivariance)
        expected = empirical.fit("gaussian", min_count=minimum)
        np.testing.assert_array_equal(empirical.fit_result["coefficients"], expected.fit_result["coefficients"])
        assert empirical.fit_result["min_count"] == minimum
        if minimum >= 100:
            assert np.any((empirical.counts > 0) & (empirical.counts < minimum))
        summary = structure.info(verbose=False)
        assert summary is not None
        assert f"Variogram lag bins: {np.count_nonzero(valid)}/10 valid (minimum {minimum} pair" in summary

        for result, threshold in ((refitted, minimum), (overridden, 1)):
            assert result.empirical_variogram is not None
            assert result.empirical_variogram.fit_result["min_count"] == threshold
            np.testing.assert_array_equal(result.empirical_variogram.counts, empirical.counts)
            np.testing.assert_array_equal(result.empirical_variogram.semivariance, empirical.semivariance)

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
        assert short.correlation.model_name == "gaussian"
        assert long.correlation.model_name == "gaussian"
        assert short.correlation.effective_range is not None and long.correlation.effective_range is not None
        assert short.magnitude.kind == "variable"
        assert long.magnitude.kind == "constant"

        # Check range order and increasing spread
        assert short.correlation.effective_range < long.correlation.effective_range
        assert predicted[1] > predicted[0]

    def test_estimate_error_structure__two_inputs_fit_correlation(self) -> None:
        """Checks that a two-raster fit recovers the same magnitude and correlation as a scaled difference proxy."""

        pytest.importorskip("skgstat")

        # Smooth spatial errors on a common grid, with a constant reference measurement
        y, x = np.mgrid[:12, :12]
        error = np.sin(x / 2) + np.cos(y / 3) + 0.1 * np.random.default_rng(17).normal(size=x.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        measured = gu.Raster.from_array(10 + error, transform, 32631, nodata=-9999)
        reference = gu.Raster.from_array(np.full_like(error, 10.0), transform, 32631, nodata=-9999)
        scaled_difference = gu.Raster.from_array(error / np.sqrt(2), transform, 32631, nodata=-9999)
        options: dict[str, Any] = {
            "components": {"spatial": {"magnitude": "constant", "correlation": "spherical"}},
            "spread_estimator": np.std,
            "n_pairs": 300,
            "fit_kwargs": {"min_count": 1},
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # Fit both routes under the same equal-precision assumption
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = measured.estimate_error_structure(reference, other_precision="same", **options)
            expected = gu.ErrorStructure.estimate(scaled_difference, **options)

        # Check the fitted component and empirical variogram on the same support
        assert result.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        distances = np.array([0.0, 10.0, 40.0])
        np.testing.assert_allclose(result.predict_correlation(distances), expected.predict_correlation(distances))
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        assert result.empirical_variogram.estimator == "matheron"
        np.testing.assert_allclose(result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance)

    def test_estimate_error_structure__in_memory_raster_accessor_variable_correlation(self) -> None:
        """Checks that an in-memory DataArray fits the same variable magnitude and correlation as a Raster."""

        pytest.importorskip("skgstat")

        # Two measurements and a quality predictor on the same 12 x 12 grid
        quality = np.broadcast_to(np.linspace(0, 1, 12), (12, 12)).copy()
        errors = (1 + quality) * np.random.default_rng(31).normal(size=quality.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        measured = gu.Raster.from_array(10 + errors, transform, 32632, nodata=-9999)
        reference = gu.Raster.from_array(np.full_like(errors, 10.0), transform, 32632, nodata=-9999)
        predictor = gu.Raster.from_array(quality, transform, 32632, nodata=-9999)
        options: dict[str, Any] = {
            "components": {"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "n_pairs": 300,
            "fit_kwargs": {"min_count": 1},
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # Fit the same scaled difference through Raster and in-memory DataArray inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = measured.estimate_error_structure(reference, predictors={"quality": predictor}, **options)
            result = measured.to_xarray().rst.estimate_error_structure(
                reference.to_xarray(), predictors={"quality": predictor.to_xarray()}, **options
            )

        # Compare the grouped magnitude table and fitted spatial correlation
        expected_magnitude = expected.components["spatial"].magnitude
        result_magnitude = result.components["spatial"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(result_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(result_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        np.testing.assert_allclose(result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance)

    @pytest.mark.parametrize("refit", [False, True], ids=["estimate", "refit"])
    @pytest.mark.parametrize(
        ("models", "variable", "independent"),
        [
            (["gaussian"], False, False),
            (["gaussian", "spherical"], False, False),
            (["gaussian", "spherical"], True, False),
            (["gaussian"], False, True),
        ],
        ids=["single", "nested", "variable", "nugget"],
    )
    def test_estimate__fitted_variogram_scale(
        self, models: list[str], variable: bool, independent: bool, refit: bool
    ) -> None:
        """Checks that estimation and refit preserve the fitted scale of empirical variogram diagnostics."""

        pytest.importorskip("skgstat")

        # Create smooth errors whose standardized variogram has a fitted sill different from one
        y, x = np.mgrid[:40, :40]
        values = np.sin(x / 4) + 0.5 * np.cos(y / 10)
        transform = Affine(10, 0, 0, 0, -10, 400)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        quality = gu.Raster.from_array(x / 39, transform, 32632, nodata=-9999)
        components: dict[str, dict[str, str | None]] = {
            f"spatial_{index}": {
                "magnitude": "heteroscedastic" if variable and index == 0 else "constant",
                "correlation": model,
            }
            for index, model in enumerate(models)
        }
        if independent:
            components["noise"] = {"magnitude": "constant", "correlation": None}

        # Estimate the components, optionally refitting the stored empirical bins
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": quality} if variable else {},
                components=components,
                n_pairs=1_000,
                n_lags=8,
                fit_kwargs={"min_count": 1},
                random_state=42,
            )
            if refit:
                structure = structure.refit(models)

        # Fit the same bins directly, before component magnitudes can normalize the fitted sill
        empirical = structure.empirical_variogram
        assert empirical is not None and empirical.model is not None
        expected = empirical.fit(models, use_nugget=independent)
        assert expected.model is not None
        assert not np.isclose(expected.model.sill, 1.0)
        assert empirical.model.sill == pytest.approx(expected.model.sill)
        assert empirical.fitted_semivariance is not None and expected.fitted_semivariance is not None
        np.testing.assert_allclose(empirical.fitted_semivariance, expected.fitted_semivariance)
        np.testing.assert_allclose(empirical.variogram(empirical.lags), expected.fitted_semivariance)

        # Component correlations still have unit sill because their magnitudes supply the variance
        for component in structure.components.values():
            if component.correlation is not None:
                assert isinstance(component.correlation, VariogramModel)
                assert component.correlation.sill == pytest.approx(1.0)

    def test_refit__uses_retained_empirical_variogram(self) -> None:
        """Checks that refit() changes the correlation model using retained empirical variogram bins."""

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
                random_state=42,
            )
            refitted = structure.refit("spherical")

        # Check the refit uses the retained empirical bins and changes the component model
        assert structure.empirical_variogram is not None
        assert refitted.empirical_variogram is not None
        np.testing.assert_array_equal(refitted.empirical_variogram.lags, structure.empirical_variogram.lags)
        refitted_model = refitted.components["spatial"].correlation
        assert isinstance(refitted_model, VariogramModel)
        assert refitted_model.model_name == "spherical"

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_estimate_error_structure__parameters(self, source_kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that dataset methods expose matching defaults and forward every estimation argument unchanged."""

        values = np.array([-2.0, -1.0, 1.0, 2.0])
        source: gu.Raster | gu.PointCloud
        if source_kind == "raster":
            source = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631)
        else:
            source = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Match parameters and defaults against the shared API
        expected = dict(inspect.signature(gu.ErrorStructure.estimate).parameters)
        expected.pop("source")
        actual = inspect.signature(source.estimate_error_structure).parameters
        assert list(actual) == list(expected)
        for name, parameter in expected.items():
            assert actual[name].default == parameter.default
            assert actual[name].annotation == parameter.annotation
            # The dataset method also accepts other positionally for existing calls
            if name != "other":
                assert actual[name].kind == parameter.kind

        # Distinct markers expose missing or reordered arguments
        options = {name: object() for name in expected}
        received: dict[str, Any] = {}
        result = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", 1)])

        def estimate(cls: type[gu.ErrorStructure], /, source: Any, **kwargs: Any) -> gu.ErrorStructure:
            """Record the source and all options received by the shared estimator."""

            received.update(source=source, options=kwargs)
            return result

        monkeypatch.setattr(gu.ErrorStructure, "estimate", classmethod(estimate))
        assert source.estimate_error_structure(**options) is result
        assert received["source"] is source
        assert received["options"] == options


class TestErrorStructureRefitting:
    """Test module for refitting stored magnitude groups and variogram bins without sampling the source again."""

    def test_refit_magnitude__manually_grouped(self) -> None:
        """Checks that refit_magnitude() fits manually supplied groups using a new sample threshold."""

        # Exclude a sparse outlying group
        statistics = pd.DataFrame(
            {"std": [1.0, 1.2, 1.8, 100], "count": [20, 20, 20, 2]},
            index=pd.Index([0.0, 1.0, 2.0, 3.0], name="angle"),
        )
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("noise", magnitude)])

        fitted = structure.refit_magnitude(quadratic_magnitude, min_count=10)

        assert fitted.predict_magnitude({"angle": 0.5}) == pytest.approx(1.05)
        updated = fitted.components["noise"].magnitude
        assert isinstance(updated, gu.ErrorMagnitude)
        assert updated.min_count == 10
        pd.testing.assert_frame_equal(updated.grouped_statistics, magnitude.grouped_statistics)

    def test_refit_magnitude__stored_groups(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that refit_magnitude() reuses grouped spreads for a custom fit and preserves the original model."""

        predictor = np.repeat(np.arange(4.0), 2)
        values = quadratic_magnitude(predictor, 1, 0.2) * np.tile([-1, 1], 4)
        source = gu.PointCloud.from_xyz(np.arange(8), np.zeros(8), values, crs=32631)
        structure = gu.ErrorStructure.estimate(
            source,
            predictors={"angle": predictor},
            bins={"angle": [-0.5, 0.5, 1.5, 2.5, 3.5]},
            components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            min_count=2,
        )
        original = structure.components["uncorrelated"].magnitude
        assert isinstance(original, gu.ErrorMagnitude)

        def fail_statistics(*args: Any, **kwargs: Any) -> Any:
            """Fail if refitting tries to estimate statistics from the point cloud again."""

            pytest.fail("Magnitude refitting must use its stored groups")

        monkeypatch.setattr(gu.PointCloud, "stats", fail_statistics)
        fitted = structure.refit_magnitude(quadratic_magnitude, fit_kwargs={"p0": [1, 0.2], "bounds": (0, np.inf)})
        repeated = fitted.refit_magnitude()
        affine = fitted.refit_magnitude("linear")

        for result in (fitted, repeated, affine):
            magnitude = result.components["uncorrelated"].magnitude
            assert isinstance(magnitude, gu.ErrorMagnitude)
            pd.testing.assert_frame_equal(magnitude.grouped_statistics, original.grouped_statistics)
            assert result.empirical_variogram is None
        assert original.model_name == "piecewise_linear"
        assert structure.predict_magnitude({"angle": 0.5}) == pytest.approx(1.1)
        assert fitted.predict_magnitude({"angle": 0.5}) == pytest.approx(1.05)
        assert repeated.predict_magnitude({"angle": 0.5}) == fitted.predict_magnitude({"angle": 0.5})

    def test_refit_magnitude__correlation_contributions(self) -> None:
        """Checks that refit_magnitude() preserves correlations and updates the model reused by refit_variogram()."""

        pytest.importorskip("skgstat")
        rows, columns = np.indices((24, 24))
        quality = columns / 23
        values = (1 + quality**2) * (np.sin(rows / 2) + np.cos(columns / 3))
        transform = Affine(10, 0, 0, 0, -10, 240)
        source = gu.Raster.from_array(values, transform, 32631)
        predictor = gu.Raster.from_array(quality, transform, 32631)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                source,
                predictors={"quality": predictor},
                components={
                    "spatial": {"magnitude": "heteroscedastic", "correlation": "gaussian"},
                    "noise": {"magnitude": "constant", "correlation": None},
                },
                bins=4,
                min_count=10,
                n_pairs=1_000,
                n_lags=8,
                fit_kwargs={"min_count": 1},
                random_state=42,
            )
            fitted = structure.refit_magnitude("linear")
            refitted = fitted.refit_variogram("exponential")

        assert fitted.empirical_variogram is structure.empirical_variogram
        assert fitted.components["noise"].magnitude == structure.components["noise"].magnitude
        for name in structure.components:
            assert fitted.components[name].correlation == structure.components[name].correlation
        magnitude = refitted.components["spatial"].magnitude
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert magnitude.model_name == "linear"

    @pytest.mark.parametrize("initial_model", ["gaussian", exponential_variogram])
    def test_refit_variogram__custom_callable(
        self, initial_model: str | Callable[..., Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that refit_variogram() fits and reuses a custom callable without sampling new pairs."""

        pytest.importorskip("skgstat")
        rows, columns = np.indices((24, 24))
        source = gu.Raster.from_array(np.sin(columns / 4) + np.cos(rows / 6), Affine(10, 0, 0, 0, -10, 240), 32631)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                source,
                components={"spatial": {"correlation": initial_model}},
                n_pairs=3_000,
                n_lags=8,
                random_state=42,
            )
        empirical = structure.empirical_variogram
        assert empirical is not None

        def fail_sampling(*args: Any, **kwargs: Any) -> Any:
            """Fail if refitting tries to draw another variogram sample."""

            pytest.fail("Variogram refitting must use its stored bins")

        monkeypatch.setattr(gu.Raster, "variogram", fail_sampling)
        fitted = structure.refit_variogram(exponential_variogram)
        repeated = fitted.refit_variogram()

        for result in (fitted, repeated):
            updated = result.empirical_variogram
            assert updated is not None and updated.model is not None
            assert updated.model.function is exponential_variogram
            assert updated.fit_result["min_count"] == empirical.fit_result["min_count"]
            columns = [name for name in empirical.to_dataframe() if name != "fitted_semivariance"]
            pd.testing.assert_frame_equal(updated.to_dataframe()[columns], empirical.to_dataframe()[columns])
        assert empirical.model is not None
        assert empirical.model.function is (None if isinstance(initial_model, str) else initial_model)


class TestErrorStructureEstimationChunked:
    """Test module for Dask/MP estimation across input types and error components."""

    def test_estimate__custom_magnitude_stays_lazy(self) -> None:
        """Checks that a custom magnitude fit preserves lazy inputs and predicts the same values as an eager fit."""

        pytest.importorskip("dask.array")
        angles = np.broadcast_to(np.tile(np.arange(4, dtype=float), 2), (8, 8)).copy()
        values = (1 + angles**2) * np.tile([-1.0, 1.0], 4)[:, None]
        transform = Affine(10, 0, 0, 0, -10, 80)
        proxy = gu.raster.from_array(values, transform, 32631)
        predictor = gu.raster.from_array(angles, transform, 32631)
        # Uneven chunks split both groups and balanced error pairs
        lazy_proxy = proxy.chunk({"y": 3, "x": 5})
        lazy_predictor = predictor.chunk({"y": 3, "x": 5})

        def model(x: Any, offset: float, scale: float) -> Any:
            """Calculate a quadratic spread from viewing angle."""

            return offset + scale * x**2

        options: dict[str, Any] = {
            "components": {"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": {"angle": [-0.5, 0.5, 1.5, 2.5, 3.5]},
            "min_count": 8,
            "magnitude_model": model,
            "random_state": 42,
        }
        assert proxy.rst.is_loaded and predictor.rst.is_loaded
        assert not lazy_proxy.rst.is_loaded and not lazy_predictor.rst.is_loaded

        eager = gu.ErrorStructure.estimate(proxy, predictors={"angle": predictor}, **options)
        lazy = gu.ErrorStructure.estimate(lazy_proxy, predictors={"angle": lazy_predictor}, **options)
        eager_map = eager.predict_magnitude({"angle": predictor}, like=proxy)
        lazy_map = lazy.predict_magnitude({"angle": lazy_predictor}, like=lazy_proxy)

        assert proxy.rst.is_loaded and predictor.rst.is_loaded
        assert not lazy_proxy.rst.is_loaded and not lazy_predictor.rst.is_loaded
        assert eager_map.rst.is_loaded and not lazy_map.rst.is_loaded
        computed = lazy_map.compute()
        assert computed.rst.is_loaded
        np.testing.assert_array_equal(computed.data, eager_map.data)
        np.testing.assert_array_equal(computed.data, 1 + angles**2)

    # Vary point representations only for Dask inputs that actually contain points
    @pytest.mark.parametrize(
        ("backend", "input_type", "as_object_type"),
        [
            ("dask", "raster", "dataarray"),
            ("dask", "point", "dataarray"),
            ("dask", "point", "geodataframe"),
            ("dask", "point-point", "dataarray"),
            ("dask", "point-point", "geodataframe"),
            ("dask", "raster-point", "dataarray"),
            ("dask", "raster-point", "geodataframe"),
            ("dask", "raster-raster", "dataarray"),
            ("multiproc", "raster", "dataarray"),
            ("multiproc", "point", "dataarray"),
            ("multiproc", "point-point", "dataarray"),
            ("multiproc", "raster-point", "dataarray"),
            ("multiproc", "raster-raster", "dataarray"),
        ],
    )
    @pytest.mark.parametrize(
        "components",
        [
            pytest.param({"uncorrelated": {"magnitude": "constant", "correlation": None}}, id="constant"),
            pytest.param({"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}}, id="variable"),
            pytest.param({"spatial": {"magnitude": "constant", "correlation": "spherical"}}, id="constant-correlation"),
            pytest.param(
                {"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
                id="variable-correlation",
            ),
            pytest.param(
                {
                    "uncorrelated": {"magnitude": "constant", "correlation": None},
                    "spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"},
                },
                id="constant-plus-variable-correlation",
            ),
        ],
    )
    def test_estimate__chunked_inputs_and_components_match_in_memory(
        self,
        as_object_type: Literal["dataarray", "geodataframe"],
        pointcloud_file: Any,
        backend: str,
        input_type: str,
        components: dict[str, dict[str, Any]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Checks that Dask/MP estimates match in-memory components for raster and point inputs.

        We vary:
         - Dask vs Multiproc (and compare to in-memory)
         - Input: single raster, single point (when directly an error proxy), or two points, two raster,
           one point and one raster (when another variable to difference with),
         - Error component definition: constant/variable magnitude, with/without correlation, and multiple components.
        """

        # 1/ Synthetic variables
        rows, columns = np.indices((12, 12))
        quality = columns / 11
        rng = np.random.default_rng(31)
        errors = (1 + quality) * (np.sin(columns / 3) + np.cos(rows / 4) + 0.1 * rng.normal(size=quality.shape))
        correlated = any(item["correlation"] is not None for item in components.values())
        variable = any(item["magnitude"] == "heteroscedastic" for item in components.values())
        # One outlier and NaN
        if not correlated:
            errors[0, 0] = 1000
            errors[0, 1] = np.nan
        reference = 10 + rows / 10 + columns / 20
        source_kind, *other_kinds = input_type.split("-")
        other_kind = other_kinds[0] if other_kinds else None
        measured = reference + errors if other_kind is not None else errors
        transform = Affine(10, 0, 0, 0, -10, 120)
        raster_source = gu.Raster.from_array(measured, transform, 32632, nodata=-9999)
        raster_reference = gu.Raster.from_array(reference, transform, 32632, nodata=-9999)
        raster_predictor = gu.Raster.from_array(quality, transform, 32632, nodata=-9999)
        # Point locations match raster pixel centers for exact gridding
        x, y = raster_source.ij2xy(rows.ravel(), columns.ravel())
        point_source = gu.PointCloud.from_xyz(x, y, measured.ravel(), crs=32632)
        point_source.gdf["quality"] = quality.ravel()
        point_reference = gu.PointCloud.from_xyz(x, y, reference.ravel(), crs=32632)
        source = raster_source if source_kind == "raster" else point_source
        if other_kind == "raster":
            other = raster_reference
        elif other_kind == "point":
            other = point_reference
        else:
            other = None
        eager_predictors = {"quality": raster_predictor if source_kind == "raster" else "quality"} if variable else None
        # We use a smaller sample size than data to also test subsampling
        options: dict[str, Any] = {
            "components": components,
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "subsample_magnitude": 60 if correlated else 1,
            "n_pairs": 300,
            "fit_kwargs": {"min_count": 1},
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # We estimate the error structure in memory for comparison later
        if correlated:
            pytest.importorskip("skgstat")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if other is None:
                expected = gu.ErrorStructure.estimate(source, predictors=eager_predictors, **options)
            else:
                expected = source.estimate_error_structure(other, predictors=eager_predictors, **options)

        # 2/ Chunked with Dask/MP, creating the same data lazy/on-file for raster or point
        # We also use an uneven chunk size relative to datasize, to check for edge behaviour
        # Use four raster tiles to limit Dask scheduling overhead
        raster_chunks = {"y": 7, "x": 8}
        chunked_other = None
        chunked_predictor = None
        input_states: list[tuple[Any, str]] = []
        if backend == "dask":
            pytest.importorskip("dask")
            if source_kind == "raster":
                chunked_source = source.to_xarray().chunk(raster_chunks)
                source_interface = chunked_source.rst
            else:
                pytest.importorskip("dask_geopandas")
                from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

                _register_dask_pointcloud_accessor()
                # Two partitions of 73 and 71 rows check the shorter final partition
                chunked_source_filename, chunked_source_chunks, chunked_source_column = pointcloud_file(source.gdf, 2)
                chunked_source = gu.open_pointcloud(
                    chunked_source_filename,
                    data_name=chunked_source_column,
                    columns="all",
                    chunks=chunked_source_chunks,
                    as_object_type=as_object_type,
                )
                source_interface = chunked_source.pc

            mp_config = None

            if other_kind == "raster":
                chunked_other = other.to_xarray().chunk(raster_chunks)
                input_states.append((chunked_other.rst, "raster"))
            elif other_kind == "point":
                pytest.importorskip("dask_geopandas")
                chunked_other_filename, chunked_other_chunks, chunked_other_column = pointcloud_file(other.gdf, 2)
                chunked_other = gu.open_pointcloud(
                    chunked_other_filename,
                    data_name=chunked_other_column,
                    columns="all",
                    chunks=chunked_other_chunks,
                    as_object_type=as_object_type,
                )

                input_states.append((chunked_other.pc, "point"))
            if variable and source_kind == "raster":
                chunked_predictor = raster_predictor.to_xarray().chunk(raster_chunks)
                input_states.append((chunked_predictor.rst, "raster"))
        else:
            # If with MP, from file inputs
            if source_kind == "raster":
                source_path = tmp_path / "source.tif"
                source.to_file(source_path)
                chunked_source = gu.Raster(source_path)
            else:
                chunked_source = source
            source_interface = chunked_source
            mp_config = MultiprocConfig(chunks=(7, 8) if source_kind == "raster" else 23)

            # Raster files must stay unloaded
            if other_kind == "raster":
                other_path = tmp_path / "other.tif"
                other.to_file(other_path)
                chunked_other = gu.Raster(other_path)
                input_states.append((chunked_other, "raster"))
            elif other_kind == "point":
                chunked_other = other
                input_states.append((chunked_other, "point"))
            if variable and source_kind == "raster":
                predictor_path = tmp_path / "quality.tif"
                raster_predictor.to_file(predictor_path)
                chunked_predictor = gu.Raster(predictor_path)
                input_states.append((chunked_predictor, "raster"))

        input_states.insert(0, (source_interface, source_kind))
        for spatial_input, kind in input_states:
            assert spatial_input.is_loaded == (backend == "multiproc" and kind == "point")
        chunked_predictors = (
            {"quality": chunked_predictor if source_kind == "raster" else "quality"} if variable else None
        )

        # For point-point, record MP point difference internally
        submitted_chunks: list[int] = []
        if backend == "multiproc" and input_type == "point-point" and not variable and not correlated:
            from geoutils.uncertainty.estimation import _difference_point_partition

            cluster = request.getfixturevalue("mp_cluster")
            mp_config = MultiprocConfig(chunks=23, cluster=cluster)
            submit = cluster.submit

            def record_submit(function: Any, *args: Any, **kwargs: Any) -> Any:
                if function is _difference_point_partition:
                    submitted_chunks.append(len(args[0]))
                return submit(function, *args, **kwargs)

            monkeypatch.setattr(cluster, "submit", record_submit)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if chunked_other is None:
                result = gu.ErrorStructure.estimate(
                    source_interface, predictors=chunked_predictors, mp_config=mp_config, **options
                )
            else:
                result = source_interface.estimate_error_structure(
                    chunked_other, predictors=chunked_predictors, mp_config=mp_config, **options
                )

        # 3/ We check exact equality across Dask/MP
        for spatial_input, kind in input_states:
            assert spatial_input.is_loaded == (backend == "multiproc" and kind == "point")
        if submitted_chunks:
            assert submitted_chunks == [23] * 6 + [5]
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(np.isfinite(errors))
        assert (
            result.fit_diagnostics["magnitude"]["sample_count"] == expected.fit_diagnostics["magnitude"]["sample_count"]
        )
        assert list(result.components) == list(expected.components)

        # Compare each component magnitude, including its grouped statistics
        test_quality = {"quality": np.array([0.2, 0.8])} if variable else None
        for name, expected_component in expected.components.items():
            component = result.components[name]
            magnitude = component.magnitude
            eager_magnitude = expected_component.magnitude
            assert isinstance(magnitude, gu.ErrorMagnitude)
            assert isinstance(eager_magnitude, gu.ErrorMagnitude)
            assert magnitude.kind == eager_magnitude.kind
            if magnitude.kind == "variable":
                assert isinstance(magnitude.grouped_statistics, pd.DataFrame)
                assert isinstance(eager_magnitude.grouped_statistics, pd.DataFrame)
                pd.testing.assert_frame_equal(magnitude.grouped_statistics, eager_magnitude.grouped_statistics)
            assert np.allclose(magnitude.predict(test_quality), eager_magnitude.predict(test_quality), equal_nan=True)

            # Compare fitted model parameters for correlated components
            correlation = component.correlation
            eager_correlation = expected_component.correlation
            if eager_correlation is None:
                assert correlation is None
            else:
                assert isinstance(correlation, VariogramModel)
                assert isinstance(eager_correlation, VariogramModel)
                assert correlation.model_name == eager_correlation.model_name
                assert correlation.effective_range == pytest.approx(eager_correlation.effective_range)
                assert correlation.partial_sill == pytest.approx(eager_correlation.partial_sill)

        # Compare the full error structure and empirical variogram
        assert np.allclose(
            result.predict_magnitude(test_quality), expected.predict_magnitude(test_quality), equal_nan=True
        )
        if not correlated:
            assert result.empirical_variogram is None
            assert expected.empirical_variogram is None
        else:
            distances = np.array([0.0, 10.0, 40.0])
            assert np.allclose(
                result.predict_correlation(distances), expected.predict_correlation(distances), equal_nan=True
            )
            assert result.empirical_variogram is not None
            assert expected.empirical_variogram is not None
            np.testing.assert_array_equal(result.empirical_variogram.counts, expected.empirical_variogram.counts)
            assert np.allclose(result.empirical_variogram.lags, expected.empirical_variogram.lags, equal_nan=True)
            assert np.allclose(
                result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance, equal_nan=True
            )


class TestErrorStructureEstimationWarnings:
    """Test module for range warnings from components contributing at least 3% of the fitted variance."""

    @pytest.mark.parametrize("long_fraction", [0.001, 0.029, 0.03, 0.04])
    @pytest.mark.parametrize(
        "ranges,messages",
        [
            ((20, 60), []),
            ((20, 95), ["longest correlation range"]),
            ((70, 95), ["ranges overlap", "longest correlation range"]),
            ((60, 20), ["ranges crossed"]),
        ],
    )
    def test_estimate__range_warnings_follow_variance_fraction(
        self, long_fraction: float, ranges: tuple[int, int], messages: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that estimate() warns about ranges only when the components contribute at least 3% of variance."""

        # Set exact fitted shares to check either side of the warning threshold
        values = np.arange(144, dtype=float).reshape(12, 12)
        proxy = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 120), 32631)
        model = VariogramModel(
            "sum",
            components=(
                VariogramModel("gaussian", effective_range=ranges[0], partial_sill=1 - long_fraction),
                VariogramModel("spherical", effective_range=ranges[1], partial_sill=long_fraction),
            ),
        )

        def fit(empirical: gu.Variogram, *args: Any, **kwargs: Any) -> gu.Variogram:
            """Return controlled fitted parameters with the measured bins."""

            # Supply enough pairs to test range warnings independently of bin filtering
            return replace(empirical, model=model, counts=np.maximum(empirical.counts, 100))

        monkeypatch.setattr(gu.Variogram, "fit", fit)
        components = {"short": {"correlation": "gaussian"}, "long": {"correlation": "spherical"}}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            result = gu.ErrorStructure.estimate(
                proxy, components=components, n_pairs=100, n_lags=6, max_lag=100, random_state=42
            )

        # Small components remain in the result without producing range warnings
        expected_messages = messages if long_fraction >= 0.03 else []
        assert len(caught) == len(expected_messages)
        for warning, message in zip(caught, expected_messages):
            assert warning.category is UserWarning
            assert message in str(warning.message)
        fractions = [component.metadata["initial_variance_fraction"] for component in result.components.values()]
        np.testing.assert_allclose(sorted(fractions), sorted([long_fraction, 1 - long_fraction]))

    @pytest.mark.parametrize("model_name", ["gaussian", "matern"])
    def test_estimate__identical_model_ranges_reorder_silently(
        self, model_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that exchanging identical model forms preserves correlation without a crossed-range warning."""

        pytest.importorskip("skgstat")
        proxy = gu.Raster.from_array(np.arange(144).reshape(12, 12), Affine(10, 0, 0, 0, -10, 120), 32631)
        smoothness = 1.5 if model_name == "matern" else None
        model = VariogramModel.sum(
            [
                VariogramModel(model_name, effective_range=60, partial_sill=0.4, smoothness=smoothness),
                VariogramModel(model_name, effective_range=20, partial_sill=0.6, smoothness=smoothness),
            ]
        )

        def fit(empirical: gu.Variogram, *args: Any, **kwargs: Any) -> gu.Variogram:
            """Return the same model forms in descending range order."""

            return replace(empirical, model=model)

        monkeypatch.setattr(gu.Variogram, "fit", fit)
        components = {name: {"correlation": model_name} for name in ["short", "long"]}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            result = gu.ErrorStructure.estimate(
                proxy, components=components, n_pairs=100, n_lags=6, max_lag=100, random_state=42
            )

        assert not caught
        fitted_ranges = []
        for component in result.components.values():
            assert isinstance(component.correlation, VariogramModel)
            fitted_ranges.append(component.correlation.effective_range)
        assert fitted_ranges == [20, 60]
        distances = np.array([0, 10, 20, 40, 60, 100])
        np.testing.assert_allclose(result.predict_correlation(distances), model.correlation(distances))

    @pytest.mark.parametrize(
        "effective_range, tail_count, warn",
        [(90, 100, False), (95, 100, False), (98, 100, True), (100, 100, True), (95, 99, True)],
    )
    def test_estimate__range_warning_uses_supported_lags(
        self, effective_range: float, tail_count: int, warn: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that range warnings require the fitted range to reach the supported lag limit."""

        pytest.importorskip("skgstat")
        proxy = gu.Raster.from_array(np.arange(144).reshape(12, 12), Affine(10, 0, 0, 0, -10, 120), 32631)
        model = VariogramModel("gaussian", effective_range=effective_range, partial_sill=1)
        lags = np.array([10, 30, 50, 70, 90, 100])

        def fit(empirical: gu.Variogram, *args: Any, **kwargs: Any) -> gu.Variogram:
            """Control the supported tail independently of the requested sampling extent."""

            return gu.Variogram(
                lags=lags,
                semivariance=model.variogram(lags),
                counts=np.array([100] * 5 + [tail_count]),
                model=model,
                attrs={"max_distance": 100},
            )

        monkeypatch.setattr(gu.Variogram, "fit", fit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            gu.ErrorStructure.estimate(proxy, n_pairs=100, n_lags=6, max_lag=100, random_state=42)

        assert len(caught) == int(warn)
        if warn:
            assert "usable lag limit" in str(caught[0].message)


class TestErrorStructureEstimationErrors:
    """Test module for invalid configurations and statistical controls in ErrorStructure.estimate()."""

    @pytest.mark.parametrize(
        "method,message",
        [("refit_magnitude", "stored variable magnitude table"), ("refit_variogram", "retained empirical variogram")],
    )
    def test_refit_methods__error_missing_statistics(self, method: str, message: str) -> None:
        """Checks an error is raised when refitting has no stored magnitude or variogram statistics."""

        structure = gu.ErrorStructure([gu.ErrorComponent("noise", 1)])

        with pytest.raises(ValueError, match=message):
            getattr(structure, method)()

    @pytest.mark.parametrize(
        "bins,categories,message",
        [
            ({"quality": 2}, {"quality": [0, 1]}, "both bins and categories"),
            (None, {"unknown": [0, 1]}, "categories must name"),
        ],
    )
    def test_estimate__error_predictor_categories(
        self, bins: dict[str, int] | None, categories: dict[str, list[int]], message: str
    ) -> None:
        """Checks an error is raised for categorical predictors with bins or unknown predictor names."""

        transform = Affine(10, 0, 0, 0, -10, 20)
        proxy = gu.raster.from_array(np.array([[-1.0, 1.0], [-2.0, 2.0]]), transform, 32631)
        quality = gu.raster.from_array(np.array([[0, 1], [0, 1]]), transform, 32631)

        with pytest.raises(ValueError, match=message):
            gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": quality},
                bins=bins,
                categories=categories,
                components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            )

    @pytest.mark.parametrize("kind", ["raster", "point"])
    def test_estimate_error_structure__error_invalid_precision(self, kind: str) -> None:
        """Checks an error is raised for an unknown second-input precision assumption."""

        # Comparable measurements on the same locations
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        if kind == "raster":
            source = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        else:
            source = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Reject assumptions with no defined variance correction
        with pytest.raises(ValueError, match="other_precision must"):
            source.estimate_error_structure(source, other_precision="unknown")

    @pytest.mark.parametrize("proxy_kind", ["raster", "point"])
    @pytest.mark.parametrize(
        "components, error_type, message",
        [
            ({}, ValueError, "at least one named"),
            ({"": {}}, TypeError, "non-empty names"),
            ({"uncorrelated": {"unknown": 1}}, ValueError, "Unknown configuration"),
            ({"uncorrelated": {"magnitude": "invalid"}}, ValueError, "magnitude must"),
            ({"uncorrelated": {"magnitude": "heteroscedastic"}}, ValueError, "requires at least one"),
            ({"uncorrelated": {"correlation": 2}}, TypeError, "variogram model name"),
        ],
    )
    def test_estimate__error_invalid_components(
        self, proxy_kind: str, components: dict[str, dict[str, Any]], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid components with raster or point inputs."""

        # Synthetic raster/point cloud as error proxy, without anything wrong with it (not the source of error)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        if proxy_kind == "raster":
            proxy = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        else:
            proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Check errors are raised for each invalid case
        with pytest.raises(error_type, match=message):
            gu.ErrorStructure.estimate(proxy, components=components)

    @pytest.mark.parametrize("proxy_kind", ["raster", "point"])
    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"fit_method": "unknown"}, NotImplementedError, "Only fit_method"),
            ({"min_count": 0}, ValueError, "min_count must"),
            ({"spread_estimator": np.min}, ValueError, "spread_estimator returned"),
            (
                {"components": {"first": {"correlation": None}, "second": {"correlation": None}}},
                ValueError,
                "Only one spatially uncorrelated component",
            ),
        ],
    )
    def test_estimate__error_invalid_fitting(
        self, proxy_kind: str, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid fitting options with raster or point inputs."""

        # Synthetic raster/point cloud as error proxy, without anything wrong with it (not the source of error)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        if proxy_kind == "raster":
            proxy = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        else:
            proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)
        default: dict[str, Any] = {"components": {"uncorrelated": {"correlation": None}}, "spread_estimator": np.std}

        # Check error for each case
        with pytest.raises(error_type, match=message):
            gu.ErrorStructure.estimate(proxy, **(default | options))

    def test_estimate__error_raster_predictor_column(self) -> None:
        """Checks an error is raised when a raster predictor is given as a point column name."""

        # Synthetic raster cloud as error proxy, without anything wrong with it (not the source of error)
        values = np.array([[-2.0, -1.0], [1.0, 2.0]])
        proxy = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)

        # Check that raster predictors cannot use point column names
        with pytest.raises(TypeError, match="Raster magnitude predictors cannot be column names"):
            gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": "quality"},
                components={"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}},
            )
