"""Tests for measuring, fitting, saving, and converting variograms."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin

import geoutils as gu
from geoutils.stats.variography import VariogramModel


@pytest.fixture(autouse=True)
def _writable_matplotlib_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep optional plotting imports inside the test workspace."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))


class TestVariogramStorage:
    """Checks the compact Variogram result and its stored distance bins.

    The methods cover optional imports, immutable arrays, serialization, pair release, and stable bin limits.
    """

    def test_importing_geoutils_does_not_load_variogram_backends(self) -> None:
        """Checks that importing GeoUtils does not import optional variogram packages."""

        # Build a fresh Python process that imports only GeoUtils
        code = (
            "import sys\n"
            "import geoutils\n"
            "assert not {'skgstat', 'gstools', 'gpytorch', 'torch'}.intersection(sys.modules)\n"
        )
        # Check inside that process that none of the optional packages was loaded
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_variogram_is_small_immutable_and_serializable(self) -> None:
        """Checks that Variogram owns read-only arrays and survives JSON and Xarray conversion."""

        # Create a complete measured and fitted result from small arrays and plain model details
        result = gu.Variogram(
            lags=np.array([1.0, 2.0]),
            semivariance=np.array([0.2, 0.5]),
            counts=np.array([10, 8]),
            semivariance_error=np.array([0.02, 0.03]),
            bin_lower_edges=np.array([0.5, 1.5]),
            bin_edges=np.array([1.5, 2.5]),
            fitted_semivariance=np.array([0.25, 0.45]),
            model=VariogramModel("gaussian", effective_range=4, partial_sill=0.8, nugget=0.1),
            estimator="matheron",
        )

        # Check that callers cannot change stored distance values in place
        with pytest.raises(ValueError, match="read-only"):
            result.lags[0] = 3

        # Convert through JSON and check the restored model and optional error values
        restored = gu.Variogram.from_dict(json.loads(json.dumps(result.to_dict())))
        assert restored.model == result.model
        assert restored.semivariance_error is not None and result.semivariance_error is not None
        assert np.array_equal(restored.semivariance_error, result.semivariance_error)
        # Check that Xarray contains every per-distance value
        assert set(restored.to_xarray().data_vars) == {
            "semivariance",
            "semivariance_error",
            "count",
            "bin_lower_edge",
            "bin_edge",
            "fitted_semivariance",
        }

    def test_from_pairs_discards_pair_data(self) -> None:
        """Checks that from_pairs() keeps per-distance results and releases individual pairs."""

        # Create twenty pairs with a constant endpoint difference of two
        values = np.column_stack((np.arange(20, dtype=float), np.arange(20, dtype=float) + 2))
        pairs = xr.Dataset(
            {"value": (("pair", "endpoint"), values), "distance": ("pair", np.linspace(1, 20, 20))},
            coords={"pair": np.arange(20), "endpoint": ["first", "second"]},
        )

        # Reduce the pairs into four log-spaced distance bins
        result = gu.Variogram.from_pairs(pairs, estimator="matheron", bins="log", n_lags=4)

        # Check the bin count, total pair count, semivariance, and released source object
        assert len(result.lags) == 4
        assert np.sum(result.counts) == 20
        assert result.backend_object is None
        assert result.semivariance == pytest.approx(np.full(4, 2.0))

    def test_from_pairs_uses_sampled_distance_limits_for_stable_bins(self) -> None:
        """Checks that from_pairs() uses requested distance limits as repeatable bin edges."""

        # Store requested limits that extend beyond the distances drawn in this pair sample
        pairs = xr.Dataset(
            {
                "value": (("pair", "endpoint"), np.column_stack((np.zeros(6), np.arange(1, 7)))),
                "distance": ("pair", np.linspace(2, 8, 6)),
            },
            attrs={"min_distance": 1.0, "max_distance": 10.0},
        )

        # Build three log-spaced bins from the stored limits
        result = gu.Variogram.from_pairs(pairs, bins="log", n_lags=3)

        # Check both outer edges and that every sampled pair enters one bin
        assert result.bin_lower_edges is not None and result.bin_edges is not None
        assert result.bin_lower_edges[0] == 1
        assert result.bin_edges[-1] == 10
        assert np.sum(result.counts) == 6


class TestVariogramEstimation:
    """Checks variogram estimation and fitting through raster and point cloud methods.

    The methods cover repeated samples, validation, model names, functions, and the shared pair API.
    """

    def test_object_variogram_aggregates_runs_and_fits_summed_model(self) -> None:
        """Checks that Raster.variogram() combines repeated samples and fits a summed model."""

        # Create a smooth raster with variation in both grid directions
        y, x = np.mgrid[:35, :35]
        raster = gu.Raster.from_array(
            np.sin(x / 4) + 0.5 * np.cos(y / 10), from_origin(0, 35, 2, 2), 32633, nodata=None
        )
        # Combine three pair samples and fit Gaussian plus spherical components
        result = raster.variogram(
            n_pairs=1_000,
            n_lags=8,
            n_runs=3,
            model=["gaussian", "spherical"],
            random_state=42,
        )

        # Check both fitted components, sampling errors, released pairs, and run count
        assert result.model is not None and result.model.model_name == "sum"
        assert [component.model_name for component in result.model.components] == ["gaussian", "spherical"]
        assert np.any(np.isfinite(result.semivariance_error))
        assert result.backend_object is None
        assert result.attrs["n_runs"] == 3

    @pytest.mark.parametrize("n_runs", [1, 3])
    def test_variogram_repetitions_match_independent_samples(self, n_runs: int) -> None:
        """Checks that repeated pair samples combine their values, counts, and sampling errors."""

        # Reproduce each pair sample separately with fixed distance bin edges
        y, x = np.mgrid[:20, :20]
        raster = gu.Raster.from_array(np.sin(x / 4) + np.cos(y / 5), from_origin(0, 20, 1, 1), 32633)
        edges = np.linspace(0, 25, 6)
        seeds = np.random.default_rng(42).integers(0, np.iinfo(np.int32).max, n_runs)
        samples = [
            gu.Variogram.from_pairs(raster.pairsample(n_pairs=500, random_state=int(seed)), bins=edges)
            for seed in seeds
        ]

        # Check the public result against the mean values and summed counts from separate samples
        result = raster.variogram(n_pairs=500, bins=edges, n_runs=n_runs, random_state=42)
        empirical = np.stack([sample.semivariance for sample in samples])
        np.testing.assert_allclose(result.semivariance, np.nanmean(empirical, axis=0), equal_nan=True)
        np.testing.assert_array_equal(result.counts, np.sum([sample.counts for sample in samples], axis=0))
        assert result.attrs["n_runs"] == n_runs

        # Check that sampling error is absent for one run and follows the usual standard error for repeated runs
        if n_runs == 1:
            assert np.all(np.isnan(result.semivariance_error))
        else:
            expected = np.nanstd(empirical, ddof=1, axis=0) / np.sqrt(np.isfinite(empirical).sum(axis=0))
            np.testing.assert_allclose(result.semivariance_error, expected, equal_nan=True)
            assert result.attrs["pair_count"] == int(result.counts.sum())

    @pytest.mark.parametrize("n_runs", [0, -1, 1.5, True])
    def test_variogram_rejects_invalid_repetitions(self, n_runs: int | float) -> None:
        """Checks that invalid repetition counts are rejected before sampling."""

        # Create a small valid raster because this option should fail before pair sampling
        raster = gu.Raster.from_array(np.arange(16, dtype=float).reshape(4, 4), from_origin(0, 4, 1, 1), 32633)

        # Reject zero, negative, fractional, and boolean run counts
        with pytest.raises(ValueError, match="n_runs must be a positive integer"):
            raster.variogram(n_runs=n_runs)

    def test_fit_accepts_short_names_and_skgstat_model_functions(self) -> None:
        """Checks that fit() accepts SciKit-GStat functions and the short model names used by xDEM."""

        # Create exact Gaussian values with the SciKit-GStat function
        skgstat = pytest.importorskip("skgstat")
        lags = np.linspace(1, 20, 12)
        empirical = gu.Variogram(
            lags=lags,
            semivariance=skgstat.models.gaussian(lags, 10, 2),
            counts=np.full(12, 100),
        )

        # Fit one function and one short string name as a summed model
        fitted = empirical.fit([skgstat.models.gaussian, "Sph"])

        # Check the standard component names stored in order
        assert fitted.model is not None
        assert [component.model_name for component in fitted.model.components] == ["gaussian", "spherical"]

    def test_pointcloud_variogram_and_advanced_pairs_share_api(self) -> None:
        """Checks that PointCloud exposes both pair samples and their reduced variogram values."""

        # Create point values that vary smoothly in both coordinate directions
        y, x = np.mgrid[:18, :18]
        pointcloud = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (np.sin(x / 3) + np.cos(y / 5)).ravel(), crs=32633)

        # Request either the individual pairs or six reduced distance bins
        pairs = pointcloud.pairsample(n_pairs=300, min_distance=1, max_distance=15, random_state=2)
        result = pointcloud.variogram(n_pairs=300, min_lag=1, max_lag=15, n_lags=6, random_state=2)

        # Check the requested sizes and that no model is fitted by default
        assert pairs.sizes == {"pair": 300, "endpoint": 2}
        assert len(result.lags) == 6
        assert result.model is None


class TestVariogramConversion:
    """Checks fitted model evaluation and conversion to supported optional packages.

    The methods cover covariance composition and equivalent GPyTorch, SciKit-GStat, and GSTools parameters.
    """

    def test_model_evaluation_and_gpytorch_parameters(self) -> None:
        """Checks that a Gaussian model gives the expected zero-distance values and GPyTorch length scale."""

        # Create a Gaussian model with no measured values on the first two coordinate columns
        result = gu.Variogram.from_model("gaussian", effective_range=12, partial_sill=3, nugget=0.2, active_dims=(0, 1))
        parameters = result.gpytorch_parameters()

        # Check the converted length scale and model values at zero distance
        assert parameters["kernel_name"] == "RBF"
        assert parameters["lengthscale"] == pytest.approx(12 / (2 * np.sqrt(2)))
        assert result.variogram(0) == 0
        assert result.correlation(0) == 1
        assert result.correlation(np.zeros((2, 3))).shape == (2, 3)

    def test_product_model_multiplies_covariances(self) -> None:
        """Checks that a product model multiplies component covariances and keeps one nugget."""

        # Combine spatial and temporal models that use different coordinate columns
        spatial = gu.Variogram.from_model("gaussian", effective_range=10, partial_sill=2, active_dims=(0, 1))
        temporal = gu.Variogram.from_model("exponential", effective_range=3, partial_sill=4, active_dims=(2,))
        combined = gu.Variogram.combine(spatial, temporal, combination="product", nugget=0.5)

        # Check the combined sill, zero-distance values, coordinate columns, and observation noise
        assert combined.model is not None
        assert combined.model.sill == 8.5
        assert combined.covariance(0) == pytest.approx(8.5)
        assert combined.variogram(0) == 0
        parameters = combined.gpytorch_parameters()
        assert [component["active_dims"] for component in parameters["components"]] == [(0, 1), (2,)]
        assert parameters["noise"] == 0.5

    def test_skgstat_estimation_can_discard_or_keep_backend(self) -> None:
        """Checks that direct SciKit-GStat estimation keeps its source object only when requested."""

        # Estimate the same coordinate values with and without keeping the SciKit-GStat object
        coordinates = np.linspace(0, 10, 40)[:, np.newaxis]
        values = np.sin(coordinates[:, 0])
        result = gu.Variogram.estimate(coordinates, values, model="gaussian", n_lags=6, normalize=False)
        kept = gu.Variogram.estimate(
            coordinates, values, model="gaussian", n_lags=6, normalize=False, keep_backend=True
        )

        # Check both storage choices and the method that releases a kept object
        assert result.backend_object is None
        assert kept.backend_object is not None
        assert kept.without_backend().backend_object is None

    @pytest.mark.parametrize("model_name,smoothness", [("gaussian", None), ("exponential", None), ("matern", 1.5)])
    def test_gstools_conversion_matches_skgstat(self, model_name: str, smoothness: float | None) -> None:
        """Checks that GSTools conversion matches SciKit-GStat values and nugget behavior."""

        # Build one model with no measured values and convert it to GSTools
        gstools = pytest.importorskip("gstools")
        skgstat = pytest.importorskip("skgstat")
        result = gu.Variogram.from_model(
            model_name, effective_range=8, partial_sill=2, nugget=0.1, smoothness=smoothness
        )
        converted = result.to_gstools(dim=1)
        lags = np.array([0.0, 0.25, 1.0, 4.0, 8.0])
        # Calculate expected semivariances with the matching SciKit-GStat function
        model_function = getattr(skgstat.models, model_name)
        expected = (
            model_function(lags, r=8, c0=2, b=0.1)
            if smoothness is None
            else model_function(lags, r=8, c0=2, s=smoothness, b=0.1)
        )

        # Check the GSTools model values and unchanged coordinate column selection
        assert isinstance(converted.model, gstools.CovModel)
        assert converted.model.vario_axis(lags) == pytest.approx(expected)
        assert converted.active_dims is None

    def test_gstools_conversion_keeps_sum_and_common_active_dims(self) -> None:
        """Checks that summed GSTools models share coordinate columns and one parent nugget."""

        # Sum two models that use the same two coordinate columns
        first = gu.Variogram.from_model("gaussian", 8, 2, active_dims=(0, 1))
        second = gu.Variogram.from_model("exponential", 20, 3, active_dims=(0, 1))
        converted = gu.Variogram.combine(first, second, nugget=0.1).to_gstools(dim=2)

        # Check the shared columns, summed variance, and parent nugget
        assert converted.active_dims == (0, 1)
        assert converted.model.var == pytest.approx(5)
        assert converted.model.nugget == pytest.approx(0.1)

    def test_gstools_rejects_component_specific_dimensions(self) -> None:
        """Checks that GSTools conversion rejects components that use different coordinate columns."""

        # Combine spatial and temporal models that select different columns
        spatial = gu.Variogram.from_model("gaussian", 8, 2, active_dims=(0, 1))
        temporal = gu.Variogram.from_model("exponential", 3, 1, active_dims=(2,))
        combined = gu.Variogram.combine(spatial, temporal)

        # Check the clear error for this unsupported conversion
        with pytest.raises(NotImplementedError, match="different dimensions"):
            combined.to_gstools(dim=3)
