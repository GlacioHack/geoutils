"""Tests for estimating error magnitudes and spatial correlations from elevation differences."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from scipy.ndimage import gaussian_filter

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.stats.variography import VariogramModel


class TestErrorStructureEstimation:
    """Test module for estimating independent and correlated error components."""

    def test_estimate__constant_independent_component(self) -> None:
        """Checks that independent errors have the standard deviation of the proxy values."""

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

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("other_precision", ["same", "negligible"])
    def test_estimate_error_structure__two_inputs(self, kind: str, other_precision: str) -> None:
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
        structure = source.estimate_error_structure(
            other,
            other_precision=other_precision,
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
        )

        # Check the magnitude and that no spatial correlation was fitted
        scale = np.sqrt(2) if other_precision == "same" else 1.0
        assert structure.predict_magnitude() == pytest.approx(np.std(errors[:3]) / scale)
        assert structure.empirical_variogram is None

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

        # Check no variogram is fitted for independent errors
        assert structure.empirical_variogram is None

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
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # Fit both routes under the same equal-precision assumption
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = measured.estimate_error_structure(reference, **options)
            expected = gu.ErrorStructure.estimate(scaled_difference, **options)

        # Check the fitted component and empirical variogram on the same support
        assert result.predict_magnitude() == pytest.approx(expected.predict_magnitude(), abs=1e-12)
        distances = np.array([0.0, 10.0, 40.0])
        np.testing.assert_allclose(result.predict_correlation(distances), expected.predict_correlation(distances))
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        np.testing.assert_allclose(result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance)

    def test_estimate_error_structure__eager_raster_accessor_variable_correlation(self) -> None:
        """Checks that an eager DataArray fits the same variable magnitude and correlation as a Raster."""

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
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # Fit the same scaled difference through Raster and eager DataArray inputs
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


class TestErrorStructureEstimationChunked:
    """Test module for chunked error structure estimation with Dask/MP."""

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_estimate_error_structure__two_inputs_match_eager(self, kind: str, backend: str, tmp_path: Path) -> None:
        """Checks that Dask/MP two-input estimates match eager estimates without loading the inputs."""

        # Independent errors on 8 x 9 cells or 37 points; uneven chunks include a shorter final block
        rng = np.random.default_rng(21)
        values = rng.normal(size=(8, 9) if kind == "raster" else 37)
        reference = np.full_like(values, 10.0)
        measured = reference + values
        if kind == "raster":
            transform = Affine(10, 0, 0, 0, -10, 80)
            source = gu.Raster.from_array(measured, transform, 32631, nodata=-9999)
            other = gu.Raster.from_array(reference, transform, 32631, nodata=-9999)
        else:
            x = np.arange(len(values), dtype=float)
            source = gu.PointCloud.from_xyz(x, np.zeros_like(x), measured, crs=32631)
            other = gu.PointCloud.from_xyz(x, np.zeros_like(x), reference, crs=32631)
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "constant", "correlation": None}},
            "spread_estimator": np.std,
        }

        # Fit in memory before splitting both measured and reference inputs
        expected = source.estimate_error_structure(other, **options)
        if backend == "dask" and kind == "raster":
            chunked_source = source.to_xarray().chunk({"x": 4, "y": 3})
            chunked_other = other.to_xarray().chunk({"x": 4, "y": 3})
            mp_config = None
            assert not chunked_source.rst.is_loaded
            assert not chunked_other.rst.is_loaded
        elif backend == "dask":
            dask_geopandas = pytest.importorskip("dask_geopandas")
            from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

            _register_dask_pointcloud_accessor()
            chunked_source = dask_geopandas.from_geopandas(source.ds, npartitions=4, sort=False).pc
            chunked_other = dask_geopandas.from_geopandas(other.ds, npartitions=4, sort=False).pc
            chunked_source.data_column = source.data_column
            chunked_other.data_column = other.data_column
            mp_config = None
            assert not chunked_source.is_loaded
            assert not chunked_other.is_loaded
        elif kind == "raster":
            source_path = tmp_path / "measured.tif"
            other_path = tmp_path / "reference.tif"
            source.to_file(source_path)
            other.to_file(other_path)
            chunked_source = gu.Raster(source_path)
            chunked_other = gu.Raster(other_path)
            mp_config = MultiprocConfig(chunks=(3, 4))
            assert not chunked_source.is_loaded
            assert not chunked_other.is_loaded
        else:
            chunked_source = source
            chunked_other = other
            mp_config = MultiprocConfig(chunks=10)

        # Compare estimates and check that Dask and file-backed inputs stay unloaded
        source_interface = chunked_source.rst if backend == "dask" and kind == "raster" else chunked_source
        result = source_interface.estimate_error_structure(chunked_other, mp_config=mp_config, **options)
        assert result.predict_magnitude() == pytest.approx(expected.predict_magnitude(), abs=1e-12)
        assert result.fit_diagnostics["magnitude"]["valid_count"] == values.size
        if kind == "raster":
            source_loaded = chunked_source.rst.is_loaded if backend == "dask" else chunked_source.is_loaded
            other_loaded = chunked_other.rst.is_loaded if backend == "dask" else chunked_other.is_loaded
            assert not source_loaded
            assert not other_loaded
        else:
            assert chunked_source.is_loaded == (backend == "multiproc")
            assert chunked_other.is_loaded == (backend == "multiproc")

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_estimate__raster_magnitude_matches_eager(self, backend: str, tmp_path: Path) -> None:
        """
        Checks that estimating error structure with a variable magnitude (but no correlation) with Dask/MP match eager
        results exactly, and inputs stay unloaded.
        """

        # 1/ We create synthetic data with varying spread, one NaN, one outlier
        rng = np.random.default_rng(31)
        quality = np.broadcast_to(np.linspace(0, 1, 12), (12, 12)).copy()
        values = 5 + (1 + quality) * rng.normal(size=quality.shape)
        values[0, 0] = 1000
        values[0, 1] = np.nan
        transform = Affine(10, 0, 0, 0, -10, 120)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor = gu.Raster.from_array(quality, transform, 32632, nodata=-9999)
        options: dict[str, Any] = {
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "random_state": 4,
        }

        # And we estimate the error magnitude in-memory first, for later comparison
        expected = gu.ErrorStructure.estimate(proxy, predictors={"quality": predictor}, **options)

        # 2/ Now with chunked backends, Dask and MP
        # We use uneven chunk size relative to data size on purpose, to test edge case there

        # If with Dask, from lazy inputs
        if backend == "dask":
            pytest.importorskip("dask")
            chunked_proxy = proxy.to_xarray().chunk({"x": 4, "y": 3})
            chunked_predictor = predictor.to_xarray().chunk({"x": 4, "y": 3})
            mp_config = None
            assert not chunked_proxy.rst.is_loaded
            assert not chunked_predictor.rst.is_loaded
        # If with MP, from file inputs
        else:
            proxy_path = tmp_path / "proxy.tif"
            predictor_path = tmp_path / "quality.tif"
            proxy.to_file(proxy_path)
            predictor.to_file(predictor_path)
            chunked_proxy = gu.Raster(proxy_path)
            chunked_predictor = gu.Raster(predictor_path)
            mp_config = MultiprocConfig(chunks=(4, 4))
            assert not chunked_proxy.is_loaded
            assert not chunked_predictor.is_loaded

        # We estimate the same error magnitude with either Dask or MP
        result = gu.ErrorStructure.estimate(
            chunked_proxy, predictors={"quality": chunked_predictor}, mp_config=mp_config, **options
        )
        # We check inputs are not loaded
        proxy_loaded = chunked_proxy.rst.is_loaded if backend == "dask" else chunked_proxy.is_loaded
        predictor_loaded = chunked_predictor.rst.is_loaded if backend == "dask" else chunked_predictor.is_loaded
        assert not proxy_loaded
        assert not predictor_loaded

        # Finally, we check exact equality of eager vs chunked
        valid = np.isfinite(values)
        magnitude = result.components["measurement"].magnitude
        eager_magnitude = expected.components["measurement"].magnitude
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert isinstance(eager_magnitude, gu.ErrorMagnitude)
        assert magnitude.scale == 1.0
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(valid)
        pd.testing.assert_frame_equal(magnitude.grouped_statistics, eager_magnitude.grouped_statistics)
        predicted = result.predict_magnitude({"quality": quality})
        expected_prediction = expected.predict_magnitude({"quality": quality})
        np.testing.assert_allclose(predicted, expected_prediction, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_estimate__raster_correlation_match_eager(self, backend: str, tmp_path: Path) -> None:
        """
        Checks that chunked estimation of error structure with variable magnitude + correlation with Dask/MP matches
        eager results exactly, and keeps inputs unloaded.
        """

        pytest.importorskip("skgstat")

        # 1/ Synthetic error proxy data
        rng = np.random.default_rng(31)
        quality = np.broadcast_to(np.linspace(0, 1, 12), (12, 12)).copy()
        values = 5 + (1 + quality) * rng.normal(size=quality.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor = gu.Raster.from_array(quality, transform, 32632, nodata=-9999)
        # We'll use magnitude from 60 of 144 cells while sampling spatial pairs from the whole grid,
        # to also test subsampling
        options: dict[str, Any] = {
            "components": {"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "subsample_magnitude": 60,
            "n_pairs": 300,
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # And we estimate the error structure in memory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = gu.ErrorStructure.estimate(proxy, predictors={"quality": predictor}, **options)

        # 2/ Same with Dask/MP, with lazy inputs or unloaded files
        # We use uneven chunk size relative to data size on purpose, to test edge case there
        if backend == "dask":
            pytest.importorskip("dask")
            chunked_proxy = proxy.to_xarray().chunk({"x": 4, "y": 3})
            chunked_predictor = predictor.to_xarray().chunk({"x": 4, "y": 3})
            mp_config = None
            assert not chunked_proxy.rst.is_loaded
            assert not chunked_predictor.rst.is_loaded
        else:
            proxy_path = tmp_path / "proxy.tif"
            predictor_path = tmp_path / "quality.tif"
            proxy.to_file(proxy_path)
            predictor.to_file(predictor_path)
            chunked_proxy = gu.Raster(proxy_path)
            chunked_predictor = gu.Raster(predictor_path)
            mp_config = MultiprocConfig(chunks=(4, 4))
            assert not chunked_proxy.is_loaded
            assert not chunked_predictor.is_loaded

        # Then we estimate the error structure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gu.ErrorStructure.estimate(
                chunked_proxy, predictors={"quality": chunked_predictor}, mp_config=mp_config, **options
            )

        # We check inputs stay lazy or unloaded, and the right output types are generated
        proxy_loaded = chunked_proxy.rst.is_loaded if backend == "dask" else chunked_proxy.is_loaded
        predictor_loaded = chunked_predictor.rst.is_loaded if backend == "dask" else chunked_predictor.is_loaded
        assert not proxy_loaded
        assert not predictor_loaded
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        assert isinstance(result.empirical_variogram.semivariance, np.ndarray)
        result_table = result.components["spatial"].magnitude
        expected_table = expected.components["spatial"].magnitude
        assert isinstance(result_table, gu.ErrorMagnitude)
        assert isinstance(expected_table, gu.ErrorMagnitude)
        assert isinstance(result_table.grouped_statistics, pd.DataFrame)
        assert isinstance(expected_table.grouped_statistics, pd.DataFrame)

        # 3/ We should have exact equality across backends, given that we used a random seed
        test_quality = {"quality": np.array([0.2, 0.8])}
        pd.testing.assert_frame_equal(result_table.grouped_statistics, expected_table.grouped_statistics)
        np.testing.assert_allclose(result.predict_magnitude(test_quality), expected.predict_magnitude(test_quality))
        np.testing.assert_array_equal(result.empirical_variogram.counts, expected.empirical_variogram.counts)
        np.testing.assert_allclose(
            result.empirical_variogram.lags, expected.empirical_variogram.lags, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance, rtol=0, atol=1e-12
        )

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_estimate__point_magnitude_matches_eager(self, backend: str) -> None:
        """Checks that Dask/MP point magnitudes match eager results while inputs keep their loading state."""

        # 1/ We create a point error proxy with varying spread, one NaN, one outlier
        rng = np.random.default_rng(31)
        quality = np.linspace(0, 1, 85)
        values = 5 + (1 + quality) * rng.normal(size=len(quality))
        values[0] = 1000
        values[1] = np.nan
        points = gu.PointCloud.from_xyz(np.arange(len(quality)), np.zeros(len(quality)), values, crs=32632)
        points.ds["quality"] = quality

        # And we estimate the error magnitude in-memory first, for later comparison
        options: dict[str, Any] = {
            "predictors": {"quality": "quality"},
            "components": {"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            "bins": 4,
            "min_count": 8,
            "spread_estimator": np.std,
            "random_state": 4,
        }
        expected = gu.ErrorStructure.estimate(points, **options)

        # 2/ Now with chunked backends, Dask and MP
        # We use uneven chunk size relative to data size on purpose, to test edge case there
        if backend == "dask":
            dask_geopandas = pytest.importorskip("dask_geopandas")
            from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

            _register_dask_pointcloud_accessor()
            source = dask_geopandas.from_geopandas(points.ds, npartitions=4, sort=False).pc
            source.data_column = points.data_column
            mp_config = None
            assert not source.is_loaded
        else:
            source = points
            mp_config = MultiprocConfig(chunks=23)
            assert source.is_loaded

        # We estimate the same error magnitude with either Dask or MP
        result = gu.ErrorStructure.estimate(source, mp_config=mp_config, **options)

        # We check Dask input stays lazy and MP input stays loaded
        assert source.is_loaded == (backend == "multiproc")

        # 3/ Finally, we check exact equality between eager and chunked
        valid = np.isfinite(values)
        magnitude = result.components["measurement"].magnitude
        eager_magnitude = expected.components["measurement"].magnitude
        assert list(result.components) == list(expected.components) == ["measurement"]
        assert result.components["measurement"].correlation is None
        assert result.empirical_variogram is None
        assert isinstance(magnitude, gu.ErrorMagnitude)
        assert isinstance(eager_magnitude, gu.ErrorMagnitude)
        assert magnitude.kind == eager_magnitude.kind == "variable"
        assert magnitude.scale == 1.0
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(valid)
        assert isinstance(magnitude.grouped_statistics, pd.DataFrame)
        pd.testing.assert_frame_equal(magnitude.grouped_statistics, eager_magnitude.grouped_statistics)
        predicted = result.predict_magnitude({"quality": quality})
        expected_prediction = expected.predict_magnitude({"quality": quality})
        np.testing.assert_array_equal(predicted, expected_prediction)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_estimate__point_correlation_matches_eager(self, backend: str) -> None:
        """Checks that Dask/MP point magnitude and correlation models match eager results."""

        pytest.importorskip("skgstat")

        # 1/ Synthetic point errors with spatial correlation
        x, y = np.meshgrid(np.arange(10, dtype=float), np.arange(10, dtype=float))
        noise = np.random.default_rng(17).normal(size=x.shape)
        quality = x / 9
        values = 5 + (1 + quality) * (np.sin(x / 3) + np.cos(y / 4) + 0.1 * noise)
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values.ravel(), crs=32632)
        points.ds["quality"] = quality.ravel()
        options: dict[str, Any] = {
            "predictors": {"quality": "quality"},
            "components": {"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "subsample_magnitude": 60,
            "n_pairs": 300,
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # We estimate the error structure in memory
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = gu.ErrorStructure.estimate(points, **options)

        # 2/ Same with Dask/MP, using lazy partitions or loaded point chunks
        # We use uneven chunk size relative to data size on purpose, to test edge case there
        if backend == "dask":
            dask_geopandas = pytest.importorskip("dask_geopandas")
            from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

            _register_dask_pointcloud_accessor()
            source = dask_geopandas.from_geopandas(points.ds, npartitions=4, sort=False).pc
            source.data_column = points.data_column
            mp_config = None
            assert not source.is_loaded
        else:
            source = points
            mp_config = MultiprocConfig(chunks=23)
            assert source.is_loaded

        # Then we estimate the error structure with Dask/MP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = gu.ErrorStructure.estimate(source, mp_config=mp_config, **options)

        # We check Dask input stays lazy and MP input unloaded
        assert source.is_loaded == (backend == "multiproc")
        result_table = result.components["spatial"].magnitude
        expected_table = expected.components["spatial"].magnitude
        assert isinstance(result_table, gu.ErrorMagnitude)
        assert isinstance(expected_table, gu.ErrorMagnitude)
        assert isinstance(result_table.grouped_statistics, pd.DataFrame)
        assert isinstance(expected_table.grouped_statistics, pd.DataFrame)
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        assert isinstance(result.empirical_variogram.semivariance, np.ndarray)

        # 3/ We check exact equality between Dask/MP and eager
        test_quality = {"quality": np.array([0.2, 0.8])}
        distances = np.array([0.0, 2.0, 5.0, 10.0])
        result_correlation = result.components["spatial"].correlation
        expected_correlation = expected.components["spatial"].correlation
        assert list(result.components) == list(expected.components) == ["spatial"]
        assert result_table.kind == expected_table.kind == "variable"
        assert result.fit_diagnostics["magnitude"]["valid_count"] == values.size
        assert isinstance(result_correlation, VariogramModel)
        assert result_correlation == expected_correlation
        pd.testing.assert_frame_equal(result_table.grouped_statistics, expected_table.grouped_statistics)
        np.testing.assert_allclose(result.predict_magnitude(test_quality), expected.predict_magnitude(test_quality))
        np.testing.assert_array_equal(result.predict_correlation(distances), expected.predict_correlation(distances))
        np.testing.assert_array_equal(result.empirical_variogram.counts, expected.empirical_variogram.counts)
        np.testing.assert_array_equal(result.empirical_variogram.lags, expected.empirical_variogram.lags)
        np.testing.assert_array_equal(
            result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance
        )


class TestErrorStructureEstimationErrors:
    """Test module for invalid configurations and statistical controls in ErrorStructure.estimate()."""

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
            ({"measurement": {"unknown": 1}}, ValueError, "Unknown configuration"),
            ({"measurement": {"magnitude": "invalid"}}, ValueError, "magnitude must"),
            ({"measurement": {"magnitude": "heteroscedastic"}}, ValueError, "requires at least one"),
            ({"measurement": {"correlation": 2}}, TypeError, "variogram model name"),
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
                "Only one independent component",
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
        default: dict[str, Any] = {"components": {"measurement": {"correlation": None}}, "spread_estimator": np.std}

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
                components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            )
