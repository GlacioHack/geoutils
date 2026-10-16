"""Tests for estimating, fitting, saving, and converting variograms."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.stats.variography import VariogramModel


@pytest.fixture(autouse=True)
def _writable_matplotlib_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep optional plotting imports inside the test workspace."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))


class TestVariogramStorage:
    """Checks the compact Variogram result and its stored distance bins.

    The methods cover optional imports, serialization, pair release, and stable bin limits.
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

    def test_to_dataframe(self) -> None:
        """Checks that the export to dataframe contains all data."""

        # We create a 2-bin variogram with optional uncertainty, bin edges, and fit parameters
        result = gu.Variogram(
            lags=np.array([1.0, 2.0]),
            semivariance=np.array([0.2, 0.5]),
            counts=np.array([10, 8]),
            semivariance_error=np.array([0.02, 0.03]),
            bin_lower_edges=np.array([0.5, 1.5]),
            bin_edges=np.array([1.5, 2.5]),
            fitted_semivariance=np.array([0.25, 0.45]),
        )

        # We export, then compare the table with input
        table = result.to_dataframe()
        assert result.semivariance_error is not None
        assert result.bin_lower_edges is not None and result.bin_edges is not None
        assert result.fitted_semivariance is not None
        assert list(table.columns) == [
            "lag",
            "semivariance",
            "count",
            "semivariance_error",
            "bin_lower_edge",
            "bin_edge",
            "fitted_semivariance",
        ]
        np.testing.assert_array_equal(
            table.to_numpy(),
            np.column_stack(
                (
                    result.lags,
                    result.semivariance,
                    result.counts,
                    result.semivariance_error,
                    result.bin_lower_edges,
                    result.bin_edges,
                    result.fitted_semivariance,
                )
            ),
        )

    def test_from_pairs_discards_pair_data(self) -> None:
        """Checks that from_pairs() keeps per-distance results and releases individual pairs."""

        pytest.importorskip("skgstat")

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

        pytest.importorskip("skgstat")

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

    def test_from_pairs__bin_edges_missing_values_and_order(self) -> None:
        """
        Checks that from_pairs() handles exact edges and pairs containing nodata without changing within-bin order.
        """

        # Mix endpoint order, exact boundaries, empty bins, and distances outside the requested range
        distances = np.array([4, 1, 2, 1.5, 6, 0.5, 7, np.nan, np.inf, 0, 3, 2.5], dtype=float)
        differences = np.arange(1, len(distances) + 1, dtype=float)
        differences[-2:] = np.nan
        pairs = xr.Dataset(
            {
                "value": (("pair", "endpoint"), np.column_stack((np.zeros(len(distances)), differences))),
                "distance": ("pair", distances),
            }
        )
        edges = np.array([1, 2, 3, 4, 6], dtype=float)

        # Use an order-sensitive estimator so sorting must preserve each bin's original pair order
        def weighted_difference(values: NDArrayNum) -> float:
            """Weight each difference by its position within the supplied bin."""
            return float(np.dot(values, np.arange(1, len(values) + 1)))

        result = gu.Variogram.from_pairs(pairs, estimator=weighted_difference, bins=edges)

        # Check independently selected right-closed bins, with the first lower edge included
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            above_lower = distances >= lower if index == 0 else distances > lower
            selected = above_lower & (distances <= upper) & np.isfinite(differences)
            assert result.counts[index] == np.count_nonzero(selected)
            if np.any(selected):
                assert result.semivariance[index] == weighted_difference(differences[selected])
                assert result.lags[index] == np.mean(distances[selected])
            else:
                assert np.isnan(result.semivariance[index])
                assert np.isnan(result.lags[index])

    def test_from_pairs__constant_distances_with_explicit_bins(self) -> None:
        """Checks that explicit bins accept pairs sharing one distance and include empty bins."""

        # Place every finite pair at one distance with an endpoint difference of two
        pairs = xr.Dataset(
            {"value": (("pair", "endpoint"), np.tile([1.0, 3.0], (4, 1))), "distance": ("pair", np.full(4, 2.0))}
        )
        result = gu.Variogram.from_pairs(pairs, estimator=np.mean, bins=[1, 2, 3])

        # The first bin includes its upper boundary; the unoccupied bin remains NaN
        np.testing.assert_array_equal(result.counts, [4, 0])
        np.testing.assert_allclose(result.semivariance, [2, np.nan], equal_nan=True)
        np.testing.assert_allclose(result.lags, [2, np.nan], equal_nan=True)


class TestVariogramEstimation:
    """Checks variogram estimation and fitting through raster and point cloud methods.

    The methods cover repeated samples, model names, functions, and the shared pair API.
    """

    def test_object_variogram_aggregates_runs_and_fits_summed_model(self) -> None:
        """Checks that Raster.variogram() combines repeated samples and fits a summed model."""

        pytest.importorskip("skgstat")

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

        pytest.importorskip("skgstat")

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
        result = gu.stats.variogram(raster, n_pairs=500, bins=edges, n_runs=n_runs, random_state=42)
        empirical = np.stack([sample.semivariance for sample in samples])
        np.testing.assert_allclose(result.semivariance, np.nanmean(empirical, axis=0), equal_nan=True)
        np.testing.assert_array_equal(result.counts, np.sum([sample.counts for sample in samples], axis=0))
        assert result.attrs["n_runs"] == n_runs

        # Check that sampling error is absent for one run and follows the usual standard error for repeated runs
        assert result.semivariance_error is not None
        if n_runs == 1:
            assert np.all(np.isnan(result.semivariance_error))
        else:
            expected = np.nanstd(empirical, ddof=1, axis=0) / np.sqrt(np.isfinite(empirical).sum(axis=0))
            np.testing.assert_allclose(result.semivariance_error, expected, equal_nan=True)
            assert result.attrs["pair_count"] == int(result.counts.sum())

    def test_variogram__repeated_explicit_bin_generator(self) -> None:
        """Checks that repeated variogram samples reuse explicit boundaries supplied as a generator."""

        # Build a raster and fixed bins that cover every sampled distance
        array = np.arange(400, dtype=float).reshape(20, 20)
        raster = gu.Raster.from_array(array, from_origin(0, 20, 1, 1), 32633)
        edges = np.linspace(0, 30, 7)
        options = {"n_pairs": 100, "n_runs": 2, "random_state": 7, "estimator": np.mean}

        # Check that every run receives the same edges even when the input can be iterated only once
        expected = raster.variogram(bins=edges, **options)
        result = raster.variogram(bins=(edge for edge in edges), **options)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_allclose(result.semivariance, expected.semivariance, equal_nan=True)

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

        pytest.importorskip("skgstat")

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

        pytest.importorskip("skgstat")

        # Create a Gaussian model with no measured values on the first two coordinate columns
        result = gu.Variogram.from_model("gaussian", effective_range=12, partial_sill=3, nugget=0.2, active_dims=(0, 1))
        parameters = result.gpytorch_parameters()

        # Check the converted length scale and model values at zero distance
        assert parameters["kernel_name"] == "RBF"
        assert parameters["lengthscale"] == pytest.approx(12 / (2 * np.sqrt(2)))
        assert result.variogram(0) == 0
        assert result.correlation(0) == 1
        assert result.correlation(np.zeros((2, 3))).shape == (2, 3)

    @pytest.mark.parametrize("model_name,smoothness", [("gaussian", None), ("exponential", None), ("matern", 1.5)])
    def test_to_gpytorch__covar(self, model_name: str, smoothness: float | None) -> None:
        """Checks that converting to GPyTorch kernel works properly to estimate covariance."""

        torch = pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        # We create a variogram
        result = gu.Variogram.from_model(
            model_name, effective_range=6, partial_sill=2, nugget=0.1, smoothness=smoothness
        )
        positions = np.array([0.0, 1.0, 3.0])
        distances = np.abs(np.subtract.outer(positions, positions))
        expected = result.covariance(distances) - np.where(distances == 0, 0.1, 0.0)

        # Convert to covariance in GPyTorch, compare to ours
        converted = result.to_gpytorch(trainable=False)
        tensor = torch.as_tensor(positions[:, None], dtype=torch.float64)
        actual = converted.kernel(tensor).to_dense().detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        assert converted.noise == pytest.approx(0.1)
        assert all(not parameter.requires_grad for parameter in converted.kernel.parameters())

    @pytest.mark.parametrize("combination", ["sum", "product"])
    def test_to_gpytorch__covar_combined(self, combination: str) -> None:
        """Checks GPyTorch export for sum/multiplied kernels."""

        torch = pytest.importorskip("torch")
        pytest.importorskip("gpytorch")

        # We combine components as sum/products
        first = gu.Variogram.from_model("gaussian", effective_range=6, partial_sill=2)
        second = gu.Variogram.from_model("exponential", effective_range=3, partial_sill=3)
        result = gu.Variogram.combine(first, second, combination=combination, nugget=0.2)
        positions = np.array([0.0, 1.0, 2.0])
        distances = np.abs(np.subtract.outer(positions, positions))
        expected = result.covariance(distances) - np.where(distances == 0, 0.2, 0.0)

        # Check equality of covariance
        converted = result.to_gpytorch(trainable=False)
        actual = converted.kernel(torch.as_tensor(positions[:, None], dtype=torch.float64)).to_dense().detach().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        assert converted.noise == pytest.approx(0.2)

    def test_plot(self) -> None:
        """Checks that Variogram.plot() runs, and contains the right axis components."""

        pyplot = pytest.importorskip("matplotlib.pyplot")
        pytest.importorskip("skgstat")

        # Synthetic variogram
        result = gu.Variogram(
            lags=np.array([1.0, 2.0, 3.0]),
            semivariance=np.array([0.3, 0.7, 1.2]),
            counts=np.array([8, 10, 9]),
            semivariance_error=np.array([0.1, 0.2, 0.1]),
            model=VariogramModel("gaussian", effective_range=4, partial_sill=2),
        )

        # We check component exists and the right data is plotted
        axes = result.plot()
        assert axes.get_xlabel() == "Lag distance"
        assert axes.get_ylabel() == "Semivariance"
        assert len(axes.lines) >= 2
        np.testing.assert_allclose(axes.lines[-1].get_ydata()[0], result.variogram(0))
        pyplot.close(axes.figure)

    def test_combine__flattens_nested_sum_and_uses_one_nugget(self) -> None:
        """Checks that combining a summed model adds its components properly, including nugget."""

        # We define 4 base models with different partial sills
        # We add a nugger to the inner sum
        first = VariogramModel("gaussian", effective_range=4, partial_sill=1)
        second = VariogramModel("exponential", effective_range=8, partial_sill=2)
        third = VariogramModel("spherical", effective_range=12, partial_sill=3)
        inner = VariogramModel.combine([first, second], combination="sum", nugget=0.25)

        # We check that combining the inner sum replaces nugget properly, and flattens the three models
        combined = VariogramModel.combine([inner, third], combination="sum", nugget=0.5)
        assert combined.model_name == "sum"
        assert combined.components == (first, second, third)
        assert combined.nugget == 0.5
        assert combined.sill == np.sum([1, 2, 3]) + 0.5

    def test_product_model_multiplies_covariances(self) -> None:
        """Checks that a product model multiplies component covariances and keeps one nugget."""

        pytest.importorskip("skgstat")

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

        pytest.importorskip("skgstat")

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

    def test_from_skgstat__summed_model(self) -> None:
        """Checks that importing a two-model SciKit-GStat models respects both ranges and sills + the nugget."""

        skgstat = pytest.importorskip("skgstat")

        # Fit two models directly with SciKit-GStat class
        coordinates = np.arange(30, dtype=float)[:, None]
        values = np.sin(coordinates[:, 0] / 3)
        backend = skgstat.Variogram(coordinates, values, model="gaussian+spherical", n_lags=6, use_nugget=True)

        # Import model
        result = gu.Variogram.from_skgstat(backend)

        # Check we imported the right partial sills, ranges and nugget
        assert result.model is not None and result.model.model_name == "sum"
        assert [component.model_name for component in result.model.components] == ["gaussian", "spherical"]
        coefficients = np.asarray(backend.cof)
        ranges = np.asarray([component.effective_range for component in result.model.components], dtype=float)
        sills = np.asarray([component.partial_sill for component in result.model.components], dtype=float)
        np.testing.assert_allclose(ranges, coefficients[[0, 2]])
        np.testing.assert_allclose(sills, coefficients[[1, 3]])
        assert result.model.nugget == pytest.approx(coefficients[4])
        assert result.backend_object is None
        assert result.fitted_semivariance is not None
        np.testing.assert_allclose(result.fitted_semivariance, backend.fitted_model(result.lags))

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

        pytest.importorskip("gstools")

        # Sum two models that use the same two coordinate columns
        first = gu.Variogram.from_model("gaussian", 8, 2, active_dims=(0, 1))
        second = gu.Variogram.from_model("exponential", 20, 3, active_dims=(0, 1))
        converted = gu.Variogram.combine(first, second, nugget=0.1).to_gstools(dim=2)

        # Check the shared columns, summed variance, and parent nugget
        assert converted.active_dims == (0, 1)
        assert converted.model.var == pytest.approx(5)
        assert converted.model.nugget == pytest.approx(0.1)


class TestVariogramErrors:
    """Test module for validation errors raised by variogram storage, estimation, and conversion."""

    @pytest.mark.skipif(find_spec("skgstat") is not None, reason="Only runs if scikit-gstat is missing.")
    def test_estimate__error_missing_scikit_gstat(self) -> None:
        """Checks that estimate() reports a missing SciKit-GStat installation."""

        # Give estimate() enough valid observations so it can proceed directly to the optional backend
        coordinates = np.arange(4, dtype=float)[:, np.newaxis]
        values = np.arange(4, dtype=float)

        # Check that GeoUtils names the installable package (its Python import is called skgstat)
        with pytest.raises(ImportError, match="Optional dependency 'scikit-gstat' required.*"):
            gu.Variogram.estimate(coordinates, values)

    @pytest.mark.skipif(find_spec("gstools") is not None, reason="Only runs if gstools is missing.")
    def test_to_gstools__error_missing_dependency(self) -> None:
        """Checks that to_gstools() reports a missing GSTools installation."""

        # Build a complete fitted model without importing any conversion backend
        variogram = gu.Variogram.from_model("gaussian", effective_range=10, partial_sill=2)

        # Check the error raised when the requested output package is unavailable
        with pytest.raises(ImportError, match="Optional dependency 'gstools' required.*"):
            variogram.to_gstools()

    @pytest.mark.skipif(find_spec("gpytorch") is not None, reason="Only runs if gpytorch is missing.")
    def test_to_gpytorch__error_missing_dependency(self) -> None:
        """Checks that to_gpytorch() reports a missing GPyTorch installation."""

        # Build a complete fitted model without importing any conversion backend
        variogram = gu.Variogram.from_model("gaussian", effective_range=10, partial_sill=2)

        # Check the error raised when the requested output package is unavailable
        with pytest.raises(ImportError, match="Optional dependency 'gpytorch' required.*"):
            variogram.to_gpytorch()

    def test_variogram__error_read_only_arrays(self) -> None:
        """Checks that callers cannot modify arrays stored by a Variogram."""

        # Create a result through the ordinary validation and array-copying path
        result = gu.Variogram(
            lags=np.array([1.0, 2.0]),
            semivariance=np.array([0.2, 0.5]),
            counts=np.array([10, 8]),
        )

        # Stored distance values remain read-only after construction
        with pytest.raises(ValueError, match="read-only"):
            result.lags[0] = 3

    @pytest.mark.parametrize(
        "options, message",
        [
            ({"lags": np.array([[1.0, 2.0]])}, "one-dimensional"),
            ({"semivariance": np.array([0.2])}, "equal lengths"),
            ({"counts": np.array([2, -1])}, "cannot be negative"),
            ({"bin_edges": np.array([1.0])}, "bin_edges.*aligned"),
        ],
    )
    def test_variogram__error_invalid_bin_arrays(self, options: dict[str, NDArrayNum], message: str) -> None:
        """Checks an error is raised for inconsistent variogram input arrays."""

        # We create valid arrays, and will update with an invalid array at once below
        arrays: dict[str, Any] = {
            "lags": np.array([1.0, 2.0]),
            "semivariance": np.array([0.2, 0.5]),
            "counts": np.array([3, 4]),
        }
        with pytest.raises(ValueError, match=message):
            gu.Variogram(**(arrays | options))

    def test_from_pairs__error_distance_dimensions(self) -> None:
        """Checks that from_pairs() rejects distances that do not provide one value per pair."""

        # Give distances an extra dimension that would otherwise broadcast against endpoint differences
        pairs = xr.Dataset(
            {"value": (("pair", "endpoint"), np.zeros((3, 2))), "distance": (("pair", "extra"), np.ones((3, 1)))}
        )

        # Reject the incompatible dimensions before estimating semivariance
        with pytest.raises(ValueError, match="Variable 'distance' in argument ``pairs`` must have dimensions"):
            gu.Variogram.from_pairs(pairs)

    @pytest.mark.parametrize("case", ["missing_values", "wrong_endpoints", "zero_distances"])
    def test_from_pairs__error_invalid_pair_values(self, case: str) -> None:
        """Checks an error is raised when pairs don't have two endpoint or positive distances."""

        # We define 3 pairs with valid endpoints/distances
        values = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        pairs = xr.Dataset({"value": (("pair", "endpoint"), values), "distance": ("pair", [1.0, 2.0, 3.0])})

        # We remove or reshape endpoint values, or make distance wrong
        if case == "missing_values":
            pairs = pairs.drop_vars("value")
            error_type: type[Exception] = TypeError
            message = "containing 'distance' and 'value'"
        elif case == "wrong_endpoints":
            pairs = pairs.isel(endpoint=[0])
            error_type = ValueError
            message = "length two"
        else:
            pairs["distance"] = ("pair", [0.0, 0.0, 0.0])
            error_type = ValueError
            message = "no finite observations with positive distance"

        # Check error
        with pytest.raises(error_type, match=message):
            gu.Variogram.from_pairs(pairs)

    @pytest.mark.parametrize("n_runs", [0, -1, 1.5, True])
    def test_variogram__error_invalid_repetitions(self, n_runs: int | float) -> None:
        """Checks that invalid repetition counts are rejected before sampling."""

        # Create a small valid raster because this option should fail before pair sampling
        raster = gu.Raster.from_array(np.arange(16, dtype=float).reshape(4, 4), from_origin(0, 4, 1, 1), 32633)

        # Reject zero, negative, fractional, and boolean run counts
        with pytest.raises(ValueError, match="Argument ``n_runs`` must be a positive integer"):
            raster.variogram(n_runs=n_runs)

    def test_gstools_conversion__error_component_specific_dimensions(self) -> None:
        """Checks that GSTools conversion rejects components that use different coordinate columns."""

        pytest.importorskip("gstools")

        # Combine spatial and temporal models that select different columns
        spatial = gu.Variogram.from_model("gaussian", 8, 2, active_dims=(0, 1))
        temporal = gu.Variogram.from_model("exponential", 3, 1, active_dims=(2,))
        combined = gu.Variogram.combine(spatial, temporal)

        # Check the clear error for this unsupported conversion
        with pytest.raises(NotImplementedError, match="different dimensions"):
            combined.to_gstools(dim=3)

    @pytest.mark.parametrize(
        "changes, message",
        [
            ({"partial_sill": -1}, "partial sill"),
            ({"model_name": "unknown"}, "Unsupported variogram model"),
            ({"effective_range": 0}, "effective range"),
            ({"nugget": -1}, "nugget"),
            ({"smoothness": 0}, "smoothness"),
            ({"shape": 0}, "shape"),
            ({"active_dims": (0, 0)}, "active_dims"),
            ({"model_name": "stable"}, "requires a shape"),
        ],
    )
    def test_variogram_model__error(self, changes: dict[str, object], message: str) -> None:
        """Checks errors for VariogramModel creation."""

        # We create valid options, and we substitute an invalid option for each case
        options: dict[str, object] = {"model_name": "gaussian", "effective_range": 2, "partial_sill": 1}
        options.update(changes)

        # Should raise error
        with pytest.raises(ValueError, match=message):
            VariogramModel(**options)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "case, message", [("one", "at least two"), ("nested", "Nested sum"), ("nugget", "omit nuggets")]
    )
    def test_variogram_model__error_invalid_composition(self, case: str, message: str) -> None:
        """Checks an error is raised for invalid variogram model composiition."""

        # We define two base models
        first = VariogramModel("gaussian", effective_range=4, partial_sill=1)
        second = VariogramModel("exponential", effective_range=8, partial_sill=2)

        if case == "one":
            # A sum needs at least two components
            components: tuple[VariogramModel, ...] = (first,)
        elif case == "nested":
            # A sum cannot directly contain another sum, combine() flattens it first
            components = (VariogramModel.combine([first, second], combination="sum"), second)
        else:
            # Component nuggets must be specified once on the parent sum
            components = (first, VariogramModel("exponential", effective_range=8, partial_sill=2, nugget=0.5))
        with pytest.raises(ValueError, match=message):
            VariogramModel("sum", components=components)

    def test_correlation__error_zerosill(self) -> None:
        """Checks that a model with zero sill has no defined correlation."""

        result = gu.Variogram.from_model("gaussian", effective_range=2, partial_sill=0)
        with pytest.raises(ValueError, match="zero sill"):
            result.correlation(1)
