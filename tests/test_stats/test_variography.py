"""Tests for estimating, fitting, saving, and converting variograms."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin

import geoutils as gu
from geoutils._dispatch import is_dask_array
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.sampling.pairsampling import _RasterPairSource
from geoutils.stats.variography import VariogramModel


def stretched_variogram(distance: Any, effective_range: float, partial_sill: float, exponent: float) -> Any:
    """Calculate a stable semivariance whose range contains about 95% of the partial sill."""

    return partial_sill * (1 - np.exp(-3 * (np.asarray(distance) / effective_range) ** exponent))


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
            "assert not {'skgstat', 'gstools', 'gpytorch', 'torch', 'flox', 'crick'}.intersection(sys.modules)\n"
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
            attrs={"crs": "EPSG:32633", "distance_units": "m"},
        )

        # Convert through JSON and check the restored model and optional error values
        restored = gu.Variogram.from_dict(json.loads(json.dumps(result.to_dict())))
        assert restored.model == result.model
        assert restored.attrs == result.attrs
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
        pairs["distance"].attrs["units"] = "km"

        # Reduce the pairs into four log-spaced distance bins
        result = gu.Variogram.from_pairs(pairs, estimator="matheron", bins="log", n_lags=4)

        # Check the bin count, total pair count, semivariance, and released source object
        assert len(result.lags) == 4
        assert np.sum(result.counts) == 20
        assert result.backend_object is None
        assert result.semivariance == pytest.approx(np.full(4, 2.0))
        assert result.attrs["distance_units"] == "km"

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


class TestVariogramPlot:
    """Test module for count panels, lag scales, and empirical and fitted semivariance or correlation plots."""

    @pytest.fixture
    def fitted_variogram(self) -> gu.Variogram:
        """Create a fitted variogram with unequal bins, sampling errors and a nugget."""

        pytest.importorskip("skgstat")
        model = VariogramModel(
            "sum",
            components=(
                VariogramModel("gaussian", effective_range=2, partial_sill=0.4),
                VariogramModel("spherical", effective_range=5, partial_sill=0.8),
            ),
            nugget=0.3,
        )
        return gu.Variogram(
            lags=np.array([0.5, 2.0, 4.5]),
            semivariance=np.array([0.2, 0.7, 1.1]),
            counts=np.array([8, 15, 6]),
            semivariance_error=np.array([0.03, 0.05, 0.08]),
            bin_lower_edges=np.array([0.0, 1.0, 3.0]),
            bin_edges=np.array([1.0, 3.0, 6.0]),
            model=model,
            estimator="matheron",
        )

    def test_plot__counts_and_log_scale(self, fitted_variogram: gu.Variogram) -> None:
        """Checks that plot() aligns pair counts with positive logarithmic lag bins on supplied axes."""

        plt = pytest.importorskip("matplotlib.pyplot")
        figure, input_axis = plt.subplots()

        axis = fitted_variogram.plot(ax=input_axis, title="Custom title")

        assert axis is input_axis and axis.get_xscale() == "log"
        count_axis = next(item for item in figure.axes if item is not axis)
        # Infer a positive first edge around the 0.5 lag for the logarithmic axis
        np.testing.assert_allclose(
            [[bar.get_x(), bar.get_width(), bar.get_height()] for bar in count_axis.patches],
            [[0.25, 0.75, 8.0], [1.0, 2.0, 15.0], [3.0, 3.0, 6.0]],
        )
        np.testing.assert_allclose(count_axis.get_xlim(), axis.get_xlim())
        assert count_axis.get_title() == "Custom title"
        lower, upper = axis.get_xlim()
        ticks = axis.get_xticks()
        np.testing.assert_allclose(ticks[(ticks >= lower) & (ticks <= upper)], [0.5, 1, 2, 5])
        assert not any("e" in label.get_text().lower() or "^" in label.get_text() for label in axis.get_xticklabels())
        plt.close(figure)

    @pytest.mark.parametrize("kind", ["semivariance", "correlation"])
    def test_plot__model_values(
        self, fitted_variogram: gu.Variogram, kind: Literal["semivariance", "correlation"]
    ) -> None:
        """Checks that semivariance and correlation plots scale model values and sampling errors by the sill."""

        plt = pytest.importorskip("matplotlib.pyplot")

        axis = fitted_variogram.plot(kind=kind, show_counts=False)

        empirical = axis.containers[0]
        fitted = next(line for line in axis.lines if line.get_label() == "Model (gaussian + spherical)")
        assert empirical.get_label() == "Empirical (Matheron)"
        values, errors = fitted_variogram.semivariance, fitted_variogram.semivariance_error
        expected_model = fitted_variogram.variogram(fitted.get_xdata())
        assert errors is not None
        # Correlation includes the nugget in its normalization
        if kind == "correlation":
            values, errors = 1 - values / 1.5, errors / 1.5
            expected_model = 1 - expected_model / 1.5
        np.testing.assert_allclose(empirical.lines[0].get_ydata(), values)
        segments = np.asarray(empirical.lines[2][0].get_segments())
        np.testing.assert_allclose(segments[:, 1, 1] - segments[:, 0, 1], 2 * errors)
        # Allow one rounding step near zero correlation
        np.testing.assert_allclose(fitted.get_ydata(), expected_model, atol=float(np.finfo(float).eps))
        assert axis.get_ylabel() == ("Correlation" if kind == "correlation" else "Semivariance")
        plt.close(axis.figure)

    @pytest.mark.parametrize(
        "attrs,unit",
        [
            ({"crs": "EPSG:32633"}, "m"),
            ({"crs": "EPSG:2263"}, "US survey ft"),
            ({"crs": "EPSG:4326"}, "°"),
            ({"crs": "EPSG:32633", "distance_units": "km"}, "km"),
            ({}, "coordinate units"),
            ({"crs": "unknown"}, "coordinate units"),
        ],
    )
    def test_plot__distance_units(self, attrs: dict[str, str], unit: str) -> None:
        """Checks that plot() labels spatial lag with explicit units or units inferred from the sampling CRS."""

        pyplot = pytest.importorskip("matplotlib.pyplot")
        result = gu.Variogram(
            lags=np.array([1.0, 2.0]), semivariance=np.array([0.2, 0.5]), counts=np.array([10, 8]), attrs=attrs
        )

        axis = result.plot(show_counts=False)

        assert axis.get_xlabel() == f"Spatial lag ({unit})"
        pyplot.close(axis.figure)


class TestVariogramEstimation:
    """Test module for variogram estimation and fitting through raster and point cloud methods."""

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    @pytest.mark.parametrize("api", ["method", "function"])
    def test_variogram__default_matheron(self, source_kind: str, api: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that variogram() defaults to Matheron, avoiding the need of crick like Dowd."""

        # Create a small grid or point cloud with the same values
        y, x = np.mgrid[:4, :4]
        values = (x + 2 * y).astype(float)
        source: gu.Raster | gu.PointCloud
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 4, 1, 1), 32633)
        else:
            source = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values.ravel(), crs=32633)

        # Run the estimator while optional packages are unavailable
        monkeypatch.setitem(sys.modules, "crick", None)
        monkeypatch.setitem(sys.modules, "skgstat", None)
        options: dict[str, Any] = {"n_pairs": 30, "n_lags": 3, "random_state": 7}
        if api == "method":
            result = source.variogram(**options)
        else:
            result = gu.stats.variogram(source, **options)
        expected = source.variogram(estimator="matheron", **options)

        # Check the estimator and exact equality
        assert result.estimator == "matheron"
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.lags, expected.lags)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)

    @pytest.mark.parametrize("object_type", ["raster", "dataarray"])
    @pytest.mark.parametrize(
        "shape,resolution,maximum",
        [
            ((12, 20), (2, 5), np.sqrt(1300)),
            ((1, 8), (3, 4), np.sqrt(148)),
            ((2, 2), (1, 1), np.sqrt(2)),
        ],
    )
    def test_variogram__default_maximum_lag(
        self, object_type: str, shape: tuple[int, int], resolution: tuple[int, int], maximum: float
    ) -> None:
        """Checks that variogram() samples and bins distances up to half the raster diagonal."""

        # Use rectangular pixels and small grids to check the physical extent
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        transform = from_origin(0, shape[0] * resolution[1], *resolution)
        constructor = gu.Raster.from_array if object_type == "raster" else gu.raster.from_array
        source = constructor(values, transform, 32633)
        interface = source if object_type == "raster" else source.rst
        options = {"n_pairs": 4, "n_lags": 3, "random_state": 42}

        result = interface.variogram(**options)
        expected = interface.variogram(max_lag=maximum, **options)

        assert result.attrs["max_distance"] == pytest.approx(maximum)
        assert result.bin_edges is not None
        assert result.bin_edges[-1] == pytest.approx(maximum)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.lags, expected.lags)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)

    @pytest.mark.parametrize("limit_name", ["max_lag", "max_distance"])
    def test_variogram__supplied_maximum_lag(self, limit_name: str) -> None:
        """Checks that variogram() uses a supplied maximum lag beyond half the raster diagonal."""

        values = np.arange(400, dtype=float).reshape(20, 20)
        source = gu.Raster.from_array(values, from_origin(0, 20, 1, 1), 32633)

        options: dict[str, Any] = {limit_name: 25}
        result = source.variogram(n_pairs=40, n_lags=4, random_state=42, **options)

        assert result.attrs["max_distance"] == 25
        assert result.bin_edges is not None
        assert result.bin_edges[-1] == 25
        assert result.lags[-1] > np.sqrt(200)

    @pytest.mark.parametrize("read_strategy", ["native", "packed_validity"])
    def test_variogram__raster_default_cache(self, read_strategy: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that raster variograms reuse accepted values and honor supplied read strategies."""

        # Create missing rows and small batches so sampling needs several endpoint reads
        values = np.arange(20 * 23, dtype=np.float32).reshape(20, 23)
        values[:4] = np.nan
        raster = gu.Raster.from_array(values, from_origin(0, 20, 1, 1), 32633, nodata=-99999)
        options: dict[str, Any] = {
            "n_pairs": 127,
            "n_lags": 6,
            "estimator": "matheron",
            "strategy": "independent",
            "batch_pairs": 47,
            "max_lag": 6,
            "random_state": 42,
        }

        # Record endpoint reads while calling the public variogram method
        reads: list[int] = []
        original = _RasterPairSource.read_pairs

        def read_pairs(
            source: _RasterPairSource, first: NDArrayNum, second: NDArrayNum, **kwargs: Any
        ) -> tuple[NDArrayNum, NDArrayNum]:
            """Record each endpoint selection and forward the original read options."""

            reads.append(len(first))
            return original(source, first, second, **kwargs)

        monkeypatch.setattr(_RasterPairSource, "read_pairs", read_pairs)
        result = raster.variogram(**options)
        default_reads = list(reads)
        reads.clear()
        expected = raster.variogram(**options, read_strategy=read_strategy)

        # Check equal estimates and the reads avoided by storing accepted values
        assert result.attrs["read_strategy"] == "reuse_values"
        assert expected.attrs.get("read_strategy", "native") == read_strategy
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.lags, expected.lags)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)
        if read_strategy == "native":
            assert reads[:-1] == default_reads
            assert reads[-1] == 127
        else:
            assert reads == [127]

    def test_variogram__runs_summed_model(self) -> None:
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
    def test_variogram__runs(self, n_runs: int) -> None:
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

    def test_variogram__fit_inputs(self) -> None:
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

    @pytest.mark.parametrize("min_count", [1, 100, 150])
    def test_fit__minimum_pair_count(self, min_count: int) -> None:
        """Checks that fit() excludes bins below min_count and preserves the empirical arrays and threshold."""

        pytest.importorskip("skgstat")
        model = VariogramModel("gaussian", effective_range=8, partial_sill=2)
        lags = np.array([0.5, 1, 2, 3, 4, 6, 8, 10], dtype=float)
        values = model.variogram(lags).copy()
        counts = np.array([0, 99, 100, 150, 200, 250, 99, 300])
        # Large errors in sparse bins must not affect the filtered fit
        values[[0, 1, 6]] = 1000
        errors = np.linspace(0.01, 0.08, len(lags))
        empirical = gu.Variogram(lags=lags, semivariance=values, counts=counts, semivariance_error=errors)
        valid = counts >= min_count
        reference = gu.Variogram(
            lags=lags[valid], semivariance=values[valid], counts=counts[valid], semivariance_error=errors[valid]
        )

        fitted = empirical.fit("gaussian", min_count=min_count)
        expected = reference.fit("gaussian", min_count=min_count)

        np.testing.assert_array_equal(fitted.lags, lags)
        np.testing.assert_array_equal(fitted.semivariance, values)
        np.testing.assert_array_equal(fitted.counts, counts)
        assert fitted.semivariance_error is not None
        np.testing.assert_array_equal(fitted.semivariance_error, errors)
        np.testing.assert_array_equal(fitted.fit_result["coefficients"], expected.fit_result["coefficients"])
        if min_count >= 100:
            np.testing.assert_allclose(fitted.fit_result["coefficients"], [8, 2], rtol=1e-7)
        restored = gu.Variogram.from_dict(json.loads(json.dumps(fitted.to_dict())))
        assert restored.fit_result["min_count"] == min_count
        assert empirical.model is None

    @pytest.mark.parametrize("use_nugget", [False, True])
    def test_fit__custom_variogram(self, use_nugget: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that custom variograms recover every fitted parameter without importing SciKit-GStat."""

        # Exact values give an independent reference for all fitted parameters
        lags = np.linspace(0.5, 20, 40)
        nugget = 0.2 if use_nugget else 0
        values = stretched_variogram(lags, 8, 2, 1.3) + nugget
        empirical = gu.Variogram(lags=lags, semivariance=values, counts=np.full(len(lags), 200))
        initial = [6, 1, 1] + ([0.1] if use_nugget else [])
        bounds = [(0.1, 20), (0, 10), (0.2, 2)] + ([(0, 1)] if use_nugget else [])

        def fail_import(*args: Any, **kwargs: Any) -> Any:
            """Fail if fitting or evaluating a custom model imports an optional backend."""

            pytest.fail("Custom variogram functions do not require an optional backend")

        monkeypatch.setattr("geoutils.stats.variography.import_optional", fail_import)
        fitted = empirical.fit(stretched_variogram, p0=initial, bounds=bounds, use_nugget=use_nugget, min_count=100)

        assert fitted.model is not None
        assert fitted.model.function is stretched_variogram
        expected = [8, 2, 1.3] + ([nugget] if use_nugget else [])
        np.testing.assert_allclose(fitted.fit_result["coefficients"], expected, rtol=1e-6)
        np.testing.assert_allclose(fitted.variogram(lags), values, rtol=1e-7)
        assert fitted.variogram(0) == 0
        assert fitted.covariance(0) == pytest.approx(2 + nugget)
        restored = pickle.loads(pickle.dumps(fitted))
        np.testing.assert_array_equal(restored.variogram(lags), fitted.variogram(lags))

    def test_fit__custom_and_named_variograms(self) -> None:
        """Checks that custom and named variograms can be fitted together as independent components."""

        pytest.importorskip("skgstat")
        lags = np.linspace(0.5, 30, 60)
        named = VariogramModel("spherical", effective_range=20, partial_sill=0.5)
        values = stretched_variogram(lags, 8, 2, 1.3) + named.variogram(lags)
        empirical = gu.Variogram(lags=lags, semivariance=values, counts=np.full(len(lags), 200))

        fitted = empirical.fit(
            [stretched_variogram, "spherical"],
            p0=[7, 1.8, 1.2, 19, 0.6],
            bounds=[(1, 30), (0, 10), (0.2, 2), (1, 30), (0, 10)],
        )

        assert fitted.model is not None
        assert fitted.model.components[0].function is stretched_variogram
        assert fitted.model.components[1].model_name == "spherical"
        np.testing.assert_allclose(fitted.variogram(lags), values, rtol=1e-6)

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

    @pytest.mark.parametrize("source_kind", ["raster", "pointcloud"])
    def test_variogram__percentile(self, source_kind: str) -> None:
        """Checks that variogram runs use the percentile properly."""

        # Create raster or point data
        pytest.importorskip("skgstat")
        y, x = np.mgrid[:9, :11]
        values = np.sin(x / 3) + np.cos(y / 5)
        source: Any
        if source_kind == "raster":
            source = gu.Raster.from_array(values, from_origin(0, 9, 1, 1), 32633)
        else:
            source = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values.ravel(), crs=32633)
        options = {"n_pairs": 100, "n_runs": 2, "bins": [0.5, 2, 4, 8, 16], "random_state": 42}

        # Independent function to check
        def percentile_90(differences: NDArrayNum) -> float:
            return float(np.percentile(differences, 90))

        # Run both
        result = source.variogram(estimator="percentile", percentile=90, method="groupwise", **options)
        expected = source.variogram(estimator=percentile_90, **options)

        # Check equality and metadata
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)
        np.testing.assert_array_equal(result.semivariance_error, expected.semivariance_error)
        assert result.attrs["percentile"] == 90
        assert result.attrs["n_runs"] == 2


class TestVariogramReduction:
    """Test module for variogram reduction."""

    def test_from_pairs__default_matheron(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that from_pairs() defaults to half the mean squared difference without optional packages."""

        # Create two bins with known endpoint differences
        differences = np.array([1.0, 3.0, 5.0, 7.0])
        pairs = xr.Dataset(
            {
                "distance": ("pair", [1.0, 2.0, 3.0, 4.0]),
                "value": (("pair", "endpoint"), np.column_stack((np.zeros(4), differences))),
            }
        )

        # Reduce the pairs with the default estimator while optional packages are unavailable
        monkeypatch.setitem(sys.modules, "crick", None)
        monkeypatch.setitem(sys.modules, "skgstat", None)
        result = gu.Variogram.from_pairs(pairs, bins=[0, 2, 4])

        # Check half the mean squared difference for [1, 3] and [5, 7]
        assert result.estimator == "matheron"
        np.testing.assert_array_equal(result.counts, [2, 2])
        np.testing.assert_array_equal(result.semivariance, [2.5, 18.5])
        assert "approximation" not in result.attrs

    @pytest.mark.parametrize("estimator", ["matheron", "cressie", "dowd", "genton", "minmax", "entropy", "percentile"])
    def test_from_pairs__skgstat_estimators(self, estimator: str) -> None:
        """Checks that named estimators match SciKit-GStat."""

        # Create two bins with different pair differences
        skgstat = pytest.importorskip("skgstat")
        if estimator in {"dowd", "percentile"}:
            pytest.importorskip("crick")
        differences = np.array([[1, 2, 4, 7], [2, 5, 9, 14]], dtype=float)
        distances = np.arange(1, 9, dtype=float)
        values = np.column_stack((np.zeros(8), differences.ravel()))
        pairs = xr.Dataset({"distance": ("pair", distances), "value": (("pair", "endpoint"), values)})

        # Calculate variograms and SciKit-GStat references
        result = gu.Variogram.from_pairs(pairs, estimator=estimator, bins=[1, 4, 8])
        estimator_function = getattr(skgstat.estimators, estimator)
        expected = [estimator_function(sample) for sample in differences]

        # Check equality
        assert result.estimator == estimator
        np.testing.assert_array_equal(result.counts, [4, 4])
        np.testing.assert_array_equal(result.lags, [2.5, 6.5])
        np.testing.assert_allclose(result.semivariance, expected)

    @pytest.mark.parametrize("estimator", ["matheron", "cressie", "minmax"])
    @pytest.mark.parametrize("backend,engine", [("geoutils", None), ("flox", None), ("flox", "numba")])
    def test_from_pairs__mergeable_variograms(
        self, estimator: str, backend: Literal["geoutils", "flox"], engine: str | None
    ) -> None:
        """Checks that mergeable variogram estimators work properly."""

        if engine == "numba":
            pytest.importorskip("numba")

        # We create pairs on near bin edges, outside bins and with NaNs
        if backend == "flox":
            pytest.importorskip("flox")
        distances = np.array([1, 2, 3, 4, 6, 9, 0, np.nan, 7, -1, np.inf], dtype=np.float32)
        differences = np.array([1, 2, 3, 4, 5, 6, 7, 8, np.nan, 9, 10], dtype=np.float32)
        pairs = xr.Dataset(
            {
                "distance": ("pair", distances),
                "value": (("pair", "endpoint"), np.column_stack((np.zeros_like(differences), differences))),
            }
        )

        # Calculate variogram estimates, and compare to direct NumPy
        result = gu.Variogram.from_pairs(
            pairs,
            estimator=estimator,
            bins=[1, 2, 4, 6, 8],
            reduction_backend=backend,
            flox_engine=engine,
        )
        expected = []
        for sample in (np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0])):
            count = len(sample)
            if estimator == "matheron":
                expected.append(np.sum(sample**2) / (2 * count))
            elif estimator == "minmax":
                expected.append((sample.max() - sample.min()) / sample.mean())
            else:
                correction = 0.457 + 0.494 / count + 0.045 / count**2
                expected.append(np.mean(np.sqrt(sample)) ** 4 / (2 * correction))

        # Check estimates, counts and empty bins
        np.testing.assert_array_equal(result.counts, [2, 2, 1, 0])
        np.testing.assert_allclose(result.lags, [1.5, 3.5, 6, np.nan], rtol=0, atol=0)
        np.testing.assert_allclose(result.semivariance, [*expected, np.nan])
        assert result.counts.dtype == np.int64
        assert result.semivariance.dtype == np.float64
        assert result.attrs["pair_count"] == 5
        assert result.backend is None

    @pytest.mark.parametrize("backend,engine", [("geoutils", None), ("flox", None), ("flox", "numba")])
    def test_from_pairs__minmax_zero_empty(self, backend: Literal["geoutils", "flox"], engine: str | None) -> None:
        """Checks edge case of minmax variogram for zero difference and empty bins."""

        if engine == "numba":
            pytest.importorskip("numba")

        # Create pair differences with zeros and nonfinite values
        if backend == "flox":
            pytest.importorskip("flox")
        differences = np.array([0, 0, 5, 5, 0, 4, 3, np.nan, np.inf])
        distances = np.arange(1, 10, dtype=float)
        values = np.column_stack((np.zeros(9), differences))
        pairs = xr.Dataset({"distance": ("pair", distances), "value": (("pair", "endpoint"), values)})

        # Calculate minmax variogram
        result = gu.Variogram.from_pairs(
            pairs, estimator="minmax", bins=[1, 2, 4, 6, 7, 9, 10], reduction_backend=backend, flox_engine=engine
        )

        # Check zero means give NaN and differences 0/4 give a ratio of 2
        np.testing.assert_array_equal(result.counts, [2, 2, 2, 1, 0, 0])
        np.testing.assert_array_equal(result.semivariance, [np.nan, 0, 2, 0, np.nan, np.nan])
        np.testing.assert_array_equal(result.lags, [1.5, 3.5, 5.5, 7, np.nan, np.nan])

    @pytest.mark.parametrize("backend", ["geoutils", "flox"])
    def test_from_pairs__count_overflow(self, backend: Literal["geoutils", "flox"]) -> None:
        """Checks that numerical overflow (happens sometimes during squaring) does not change valid pair counts."""

        if backend == "flox":
            pytest.importorskip("flox")

        # Create differences for which the squares will overflow float64
        values = np.array([[0, 1e200], [0, 1e200]])
        pairs = xr.Dataset({"distance": ("pair", [1.0, 10.0]), "value": (("pair", "endpoint"), values)})

        # Calculate Matheron estimates
        with np.errstate(over="ignore"):
            result = gu.Variogram.from_pairs(pairs, estimator="matheron", bins=[0.5, 2.0], reduction_backend=backend)

        # Check we still have the proper pair count, despite overflow (inf semivariance)
        np.testing.assert_array_equal(result.counts, [1])
        np.testing.assert_array_equal(result.lags, [1])
        np.testing.assert_array_equal(result.semivariance, [np.inf])

    @pytest.mark.parametrize("percentile", [0, 25, 50, 90, 100])
    def test_from_pairs__percentile(self, percentile: float) -> None:
        """Checks that the percentile estimator matches NumPy."""

        # Create two bins with different pair differences
        pytest.importorskip("skgstat")
        differences = np.array([[1, 2, 4, 7], [2, 5, 9, 14]], dtype=float)
        values = np.column_stack((np.zeros(8), differences.ravel()))
        pairs = xr.Dataset({"distance": ("pair", np.arange(1, 9)), "value": (("pair", "endpoint"), values)})

        # Calculate exact percentiles and a NumPy reference
        result = gu.Variogram.from_pairs(
            pairs, estimator="percentile", percentile=percentile, bins=[1, 4, 8, 10], method="groupwise"
        )
        expected = np.r_[np.percentile(differences, percentile, axis=1), np.nan]

        # Check equality, empty bins and serialization
        np.testing.assert_array_equal(result.semivariance, expected)
        np.testing.assert_array_equal(result.counts, [4, 4, 0])
        assert result.attrs["percentile"] == percentile
        assert gu.Variogram.from_dict(result.to_dict()).attrs["percentile"] == percentile

    @pytest.mark.parametrize(
        "estimator,percentile", [("dowd", None)] + [("percentile", value) for value in (0, 25, 50, 90, 100)]
    )
    def test_from_pairs__approximate_estimators(self, estimator: str, percentile: float | None) -> None:
        """Checks that t-digest estimators approximate exact results."""

        # Create lognormal values and an exact reference
        pytest.importorskip("crick")
        pytest.importorskip("flox")
        rng = np.random.default_rng(42)
        distances = rng.uniform(1, 8, 10_000)
        values = rng.lognormal(size=(len(distances), 2))
        pairs = xr.Dataset({"distance": ("pair", distances), "value": (("pair", "endpoint"), values)})
        options: dict[str, Any] = {"estimator": estimator, "bins": [1, 3, 8, 10]}
        if percentile is not None:
            options["percentile"] = percentile
        exact = gu.Variogram.from_pairs(pairs, method="groupwise", **options)

        # Calculate t-digest estimates
        result = gu.Variogram.from_pairs(
            pairs, **options, reduction_backend="flox", method="map-reduce", tdigest_compression=200
        )

        # We allow a less than 2% approximation error
        np.testing.assert_array_equal(result.counts, exact.counts)
        np.testing.assert_allclose(result.lags, exact.lags)
        np.testing.assert_allclose(result.semivariance, exact.semivariance, rtol=0.02)
        if percentile is not None:
            assert result.attrs["percentile"] == percentile
        assert result.attrs["approximation"] == {"method": "tdigest", "compression": 200}
        assert "approximation" not in exact.attrs
        assert np.isnan(result.semivariance[-1])
        restored = gu.Variogram.from_dict(result.to_dict())
        assert restored.attrs["approximation"] == result.attrs["approximation"]


class TestVariogramConversion:
    """
    Checks fitted model evaluation and conversion to supported optional packages.

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

        pytest.importorskip("skgstat")
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

        pytest.importorskip("skgstat")
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
        assert axes.get_xlabel() == "Spatial lag (coordinate units)"
        assert axes.get_ylabel() == "Semivariance"
        assert len(axes.lines) >= 2

        # Compare the fitted curve at its plotted distances, which are positive on the default logarithmic axis
        fitted_curve = axes.lines[-1]
        np.testing.assert_allclose(fitted_curve.get_ydata(), result.variogram(fitted_curve.get_xdata()))
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


class TestVariogramReductionChunked:
    """Test module for chunked variogram reduction."""

    def test_variogram__default_maximum_lag(self) -> None:
        """Checks that lazy raster variograms use the same default maximum lag and values as eager rasters."""

        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")

        # Shorter final chunks check that the lag limit uses the full grid dimensions
        values = np.arange(143, dtype=float).reshape(11, 13)
        source = gu.raster.from_array(da.from_array(values, chunks=(4, 5)), from_origin(0, 11, 2, 3), 32633)
        eager = source.compute()
        assert not source.rst.is_loaded
        assert eager.rst.is_loaded

        # Select the same pairs across chunk layouts
        options = {"n_pairs": 60, "n_lags": 6, "random_state": 42, "strategy": "independent"}
        with dask.config.set(scheduler="synchronous"):
            result = source.rst.variogram(**options)
        expected = eager.rst.variogram(**options)

        assert not source.rst.is_loaded
        assert is_dask_array(source.data)
        assert isinstance(result.semivariance, np.ndarray)
        assert result.attrs["max_distance"] == expected.attrs["max_distance"]
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.lags, expected.lags)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)

    @pytest.mark.parametrize(
        "backend,execution", [("geoutils", "dask"), ("flox", "dask"), ("geoutils", "mp"), ("geoutils", "lazy_mp")]
    )
    @pytest.mark.parametrize("estimator", ["matheron", "cressie", "minmax", "dowd", "percentile"])
    @pytest.mark.parametrize("bins", [[1, 3, 7, 12, 15], "uniform"])
    def test_from_pairs__loading_laziness(
        self,
        backend: Literal["geoutils", "flox"],
        execution: str,
        estimator: str,
        bins: Any,
    ) -> None:
        """Checks that chunked variogram reductions avoid gathering all pairs."""

        # Create lazy pair blocks with a NaN
        if estimator in {"dowd", "percentile"}:
            pytest.importorskip("crick")
        if backend == "flox":
            pytest.importorskip("flox")
        distances = np.tile(np.arange(1, 12, dtype=np.float32), 7)
        values = np.column_stack((distances / 3, np.sin(distances))).astype(np.float32)
        values[5, 1] = np.nan
        eager = xr.Dataset({"distance": ("pair", distances), "value": (("pair", "endpoint"), values)})
        options: dict[str, Any] = {"estimator": estimator, "bins": bins, "n_lags": 3, "reduction_backend": backend}
        if estimator in {"dowd", "percentile"}:
            options.update(method="map-reduce", tdigest_compression=200)
        if estimator == "percentile":
            options["percentile"] = 90
        expected = gu.Variogram.from_pairs(eager, **options)
        source = eager.copy()
        scheduler: Any = nullcontext()
        callback: Any = nullcontext()

        # Split lazy endpoints into batches with a shorter final batch to reveal missing or duplicated pairs
        if execution != "mp":
            dask = pytest.importorskip("dask")
            da = pytest.importorskip("dask.array")
            from dask.callbacks import Callback

            endpoint_blocks = [
                da.from_delayed(
                    dask.delayed(np.copy)(values[start : start + 13]),
                    shape=(len(values[start : start + 13]), 2),
                    dtype=np.float32,
                )
                for start in range(0, len(values), 13)
            ]
            source["value"] = (("pair", "endpoint"), da.concatenate(endpoint_blocks))
            source["distance"] = ("pair", da.from_array(distances, chunks=11))
            scheduler = dask.config.set(scheduler="synchronous")

        # Prevent gathering all pairs during Dask reductions and lazy reads inside MP batches
        def check_task(key: Any, value: Any, *args: Any) -> None:
            """Fail if a task gathers all endpoint values or pair differences."""

            if isinstance(value, np.ndarray):
                assert value.shape not in {(len(values),), values.shape}

        if execution != "mp":
            callback = Callback(posttask=check_task)
        config = MultiprocConfig(chunks=17) if execution != "dask" else None
        with scheduler, callback:
            result = gu.Variogram.from_pairs(source, **options, mp_config=config)

        # Check equality and loading, with a bit of rounding from summing chunks
        assert is_dask_array(source.value.data) == (execution != "mp")
        assert is_dask_array(source.distance.data) == (execution != "mp")
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_allclose(result.lags, expected.lags)
        np.testing.assert_allclose(result.semivariance, expected.semivariance)

    @pytest.mark.parametrize("arrays", ["values_lazy", "distances_lazy", "different_chunks"])
    @pytest.mark.parametrize("estimator", ["matheron", "cressie", "minmax"])
    @pytest.mark.parametrize("backend,engine", [("geoutils", None), ("flox", None), ("flox", "numba")])
    def test_from_pairs__aligned_pair_columns(
        self, arrays: str, estimator: str, backend: Literal["geoutils", "flox"], engine: str | None
    ) -> None:
        """Checks that differently chunked pair columns give equivalent results."""

        if engine == "numba":
            pytest.importorskip("numba")

        # Create pairs with NaNs and a zero distance
        da = pytest.importorskip("dask.array")

        if backend == "flox":
            pytest.importorskip("flox")
        distances = np.tile([1, 2, 3, 5, 8, 9], 9).astype(np.float32)
        values = np.column_stack((np.zeros_like(distances), np.sqrt(distances)))
        values[:7, 1] = np.nan
        distances[11] = 0
        eager = xr.Dataset({"distance": ("pair", distances), "value": (("pair", "endpoint"), values)})
        pairs = eager.copy()
        if arrays != "values_lazy":
            pairs["distance"] = ("pair", da.from_array(distances, chunks=7))
        if arrays != "distances_lazy":
            pairs["value"] = (("pair", "endpoint"), da.from_array(values, chunks=(11, 1)))
        options: dict[str, Any] = {"estimator": estimator, "bins": [1, 3, 8, 10, 12]}
        expected = gu.Variogram.from_pairs(eager, **options)

        # Calculate and check equality and loading
        result = gu.Variogram.from_pairs(pairs, **options, reduction_backend=backend, flox_engine=engine)
        assert isinstance(pairs.distance.data, da.Array) == (arrays != "values_lazy")
        assert isinstance(pairs.value.data, da.Array) == (arrays != "distances_lazy")
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_allclose(result.lags, expected.lags)
        np.testing.assert_allclose(result.semivariance, expected.semivariance)

    @pytest.mark.parametrize(
        "estimator,backend",
        [
            ("matheron", "geoutils"),
            ("matheron", "flox"),
            ("cressie", "geoutils"),
            ("cressie", "flox"),
            ("minmax", "geoutils"),
            ("minmax", "flox"),
            ("tdigest_dowd", "flox"),
            ("tdigest_percentile", "flox"),
        ],
    )
    def test_from_pairs__disk_pair_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, estimator: str, backend: str
    ) -> None:
        """Checks that variogram reduction reads only the required disk columns."""

        # Write a raster with NaNs for disk pair sampling
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")

        from geoutils.sampling import pairsampling

        if backend == "flox":
            pytest.importorskip("flox")
            if estimator.startswith("tdigest_"):
                pytest.importorskip("crick")
        values = np.sin(np.arange(41 * 53, dtype=np.float32)).reshape(41, 53)
        values[2:7, 4:11] = np.nan
        transform = from_origin(0, 41, 1, 1)
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        filename = tmp_path / "source.tif"
        eager.to_file(filename)
        source = gu.Raster(filename)
        sample_options: dict[str, Any] = {
            "n_pairs": 701,
            "strategy": "independent",
            "batch_pairs": 97,
            "random_state": 42,
            "max_distance": 35,
        }
        reduction_options: dict[str, Any] = {
            "estimator": estimator.removeprefix("tdigest_"),
            "reduction_backend": backend,
            "bins": [1, 3, 8, 15, 35],
        }
        if estimator.startswith("tdigest_"):
            reduction_options.update(method="map-reduce", tdigest_compression=200)
        if estimator == "tdigest_percentile":
            reduction_options["percentile"] = 90
        expected_pairs = eager.pairsample(**sample_options)
        expected = gu.Variogram.from_pairs(expected_pairs, **reduction_options)
        pairs = source.pairsample(
            **sample_options,
            mp_config=MultiprocConfig(chunks=(7, 11)),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=37,
        )
        columns = []
        original_read = pairsampling._PairFileReader.__call__

        # Record which saved columns the reduction actually reads
        def record_column(reader: Any, filename: str, column: str, dtype: Any) -> Any:
            columns.append(column)
            return original_read(reader, filename, column, dtype)

        # Record disk reads
        assert not source.is_loaded
        assert all(isinstance(column.data, da.Array) for column in pairs.data_vars.values())
        monkeypatch.setattr(pairsampling._PairFileReader, "__call__", record_column)
        with dask.config.set(scheduler="synchronous"):
            result = gu.Variogram.from_pairs(pairs, **reduction_options)
        assert set(columns) == {"value", "distance"}
        assert not source.is_loaded
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_allclose(result.lags, expected.lags)

        # Allow rounding and small t-digest errors
        if estimator.startswith("tdigest_"):
            np.testing.assert_allclose(result.semivariance, expected.semivariance, rtol=0.02)
        else:
            np.testing.assert_allclose(result.semivariance, expected.semivariance)

    @pytest.mark.parametrize(
        "read_strategy,estimator,backend,storage",
        [
            ("native", "matheron", "geoutils", "memory"),
            ("reuse_values", "cressie", "flox", "memory"),
            ("packed_validity", "minmax", "geoutils", "memory"),
            ("native", "cressie", "geoutils", "disk"),
            ("reuse_values", "minmax", "geoutils", "disk"),
            ("packed_validity", "matheron", "geoutils", "disk"),
        ],
    )
    def test_variogram__runs(
        self, tmp_path: Path, storage: str, read_strategy: str, estimator: str, backend: str
    ) -> None:
        """Checks that chunked repeated variograms match in-memory results."""

        # Create a chunked raster with NaNs
        pytest.importorskip("skgstat")
        import gc

        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")

        if backend == "flox":
            pytest.importorskip("flox")
        values = np.sin(np.arange(41 * 53, dtype=np.float32)).reshape(41, 53)
        values[2:7, 4:11] = np.nan
        transform = from_origin(0, 41, 1, 1)
        source = gu.DataArrayRasterAccessor.from_array(da.from_array(values, chunks=(7, 11)), transform, 32633).rst
        eager = gu.Raster.from_array(values, transform, 32633, nodata=-99999)
        options: dict[str, Any] = {
            "n_pairs": 301,
            "n_runs": 3,
            "strategy": "independent",
            "batch_pairs": 97,
            "estimator": estimator,
            "random_state": 42,
            "bins": [1, 3, 8, 15, 35],
            "max_distance": 35,
            "model": "spherical",
        }
        expected = eager.variogram(**options)
        output_options: dict[str, Any] = {}
        if storage == "disk":
            filename = tmp_path / "source.tif"
            eager.to_file(filename)
            source = gu.Raster(filename)
            output_options = {"output_directory": tmp_path, "mp_config": MultiprocConfig(chunks=(7, 11))}

        # Run repeated variograms and fit a model
        with dask.config.set(scheduler="synchronous"):
            result = source.variogram(
                **options,
                **output_options,
                read_strategy=read_strategy,
                output_storage=storage,
                output_chunks=37,
                reduction_backend=backend,
            )

        # Allow rounding from summing chunks and fitting
        assert not source.is_loaded
        if storage == "memory":
            assert isinstance(source.data, da.Array)
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_allclose(result.lags, expected.lags)
        np.testing.assert_allclose(result.semivariance, expected.semivariance)
        np.testing.assert_allclose(result.semivariance_error, expected.semivariance_error)
        np.testing.assert_allclose(result.fitted_semivariance, expected.fitted_semivariance, rtol=1e-6)
        assert result.attrs["n_runs"] == 3
        assert result.attrs["reduction_backend"] == backend
        assert result.backend == "scikit-gstat models/scipy fit"
        gc.collect()
        assert not list(tmp_path.glob("geoutils-pair-output-*"))

    @pytest.mark.parametrize("percentile", [0, 25, 90, 100])
    @pytest.mark.parametrize("chunks", [7, 13])
    def test_from_pairs__percentile_loading_laziness(self, percentile: float, chunks: int) -> None:
        """Checks that lazy percentile reductions match in-memory results."""

        # Create constant bin differences with an empty bin and an invalid first chunk
        pytest.importorskip("crick")
        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")

        distances = np.tile([1.0, 3.0, 5.0], 17)
        differences = distances.copy()
        differences[:13] = np.nan
        pairs = xr.Dataset(
            {
                "distance": ("pair", distances),
                "value": (("pair", "endpoint"), np.column_stack((np.zeros(len(distances)), differences))),
            }
        )
        lazy = pairs.chunk({"pair": chunks, "endpoint": 1})
        options: dict[str, Any] = {
            "estimator": "percentile",
            "percentile": percentile,
            "bins": [1, 2, 4, 6, 8],
            "reduction_backend": "flox",
            "method": "map-reduce",
        }
        expected = gu.Variogram.from_pairs(pairs, **options)

        # Calculate lazy percentiles
        result = gu.Variogram.from_pairs(lazy, **options)

        # Check exact equality and loading
        assert isinstance(lazy.value.data, da.Array)
        assert isinstance(lazy.distance.data, da.Array)
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.lags, expected.lags)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)
        np.testing.assert_array_equal(result.semivariance, [1, 3, 5, np.nan])

    @pytest.mark.parametrize("backend", ["geoutils", "flox"])
    def test_from_pairs__distributed_dowd(self, backend: Literal["geoutils", "flox"]) -> None:
        """Checks that distributed Dowd estimates match in-memory results."""

        # Create constant bin differences with an invalid first chunk
        pytest.importorskip("crick")
        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")
        pytest.importorskip("distributed")
        from distributed import Client, LocalCluster

        distances = np.tile([1.0, 3.0, 5.0], 17)
        differences = distances.copy()
        differences[:7] = np.nan
        pairs = xr.Dataset(
            {
                "distance": ("pair", distances),
                "value": (("pair", "endpoint"), np.column_stack((np.zeros(len(distances)), differences))),
            }
        )
        eager = gu.Variogram.from_pairs(pairs, estimator="dowd", bins=[1, 2, 4, 6, 8])
        lazy = pairs.chunk({"pair": 7, "endpoint": 1})
        with LocalCluster(n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None) as cluster:
            with Client(cluster):
                result = gu.Variogram.from_pairs(
                    lazy,
                    estimator="dowd",
                    bins=[1, 2, 4, 6, 8],
                    reduction_backend=backend,
                    method="map-reduce",
                )

        # Check equality and loading
        assert isinstance(lazy.value.data, da.Array)
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, eager.counts)
        np.testing.assert_array_equal(result.lags, eager.lags)
        np.testing.assert_array_equal(result.semivariance, eager.semivariance)

    @pytest.mark.parametrize("output_chunks", [None, 17, np.int64(17)])
    def test_variogram__multiprocess_pair_batches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, output_chunks: int | np.int64 | None
    ) -> None:
        """Checks that raster MP reduction uses pair batches without changing the source's spatial tiles."""

        # Write a raster with missing cells and rectangular source tiles
        values = np.arange(99, dtype=float).reshape(9, 11)
        values[1:3, 2:4] = np.nan
        eager = gu.Raster.from_array(values, from_origin(0, 9, 1, 1), 32633, nodata=-99999)
        filename = tmp_path / "pairs.tif"
        eager.to_file(filename)
        source = gu.Raster(filename)
        config = MultiprocConfig(chunks=(4, 3))

        # Reproduce the same draws eagerly, including a shorter final batch when chunks contain 17 pairs
        options: dict[str, Any] = {
            "n_pairs": 71,
            "n_lags": 5,
            "estimator": "matheron",
            "strategy": "independent",
            "random_state": 42,
        }
        if output_chunks is not None:
            options["output_chunks"] = output_chunks
        expected = eager.variogram(**options)
        batch_sizes: list[int] = []
        original_submit = config.cluster.submit

        def submit(function: Any, *arguments: Any, **kwargs: Any) -> Any:
            """Record pair batch sizes while submitting the original worker functions."""

            if function.__name__ == "_reduce_block":
                batch_sizes.append(arguments[0][0].shape[0])
            return original_submit(function, *arguments, **kwargs)

        # Run source reads and pair reduction through the same basic cluster
        monkeypatch.setattr(config.cluster, "submit", submit)
        assert not source.is_loaded
        result = source.variogram(**options, mp_config=config)

        # Check the full sample, unchanged source tiles and exact eager result
        expected_batches = [71] if output_chunks is None else [17, 17, 17, 17, 3]
        assert batch_sizes == expected_batches
        assert config.chunks == (4, 3)
        assert not source.is_loaded
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, expected.counts)

        # Float64 summation order can differ when the same pairs are split into several batches
        np.testing.assert_allclose(result.lags, expected.lags, rtol=2e-13)
        np.testing.assert_allclose(result.semivariance, expected.semivariance, rtol=2e-13)

    def test_variogram__multiprocess_sampling_and_reduction(self, tmp_path: Path, mp_cluster: MpCluster) -> None:
        """Checks that multiprocessing variograms leave file inputs unloaded."""

        # Write and reopen a raster with NaNs
        pytest.importorskip("crick")
        values = np.arange(99, dtype=float).reshape(9, 11)
        values[1:3, 2:4] = np.nan
        eager = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 9), 32631, nodata=np.nan)
        filename = tmp_path / "pairs.tif"
        eager.to_file(filename)
        source = gu.Raster(filename)
        options: dict[str, Any] = {
            "n_pairs": 127,
            "sampling": "random_xy",
            "random_state": 42,
            "estimator": "dowd",
            "bins": [1, 3, 6, 10, 15],
            "method": "map-reduce",
            "tdigest_compression": 200,
        }
        expected = eager.variogram(**options)

        # Sample and reduce pairs with the same process pool
        assert not source.is_loaded
        pytest.importorskip("dask.array")
        result = source.variogram(
            **options,
            mp_config=MultiprocConfig(chunks=(5, 7), cluster=mp_cluster),
            output_storage="disk",
            output_directory=tmp_path,
            output_chunks=17,
        )

        # Check equality and loading
        assert not source.is_loaded
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_allclose(result.lags, expected.lags)
        np.testing.assert_allclose(result.semivariance, expected.semivariance, rtol=0.02)

    @pytest.mark.parametrize("bins", [[0.5, 2], [2, 3]])
    def test_from_pairs__multiprocess_single_read(self, bins: list[float]) -> None:
        """Checks that multiprocessing reads each pair batch once."""

        # Create three lazy pair blocks and record reads
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")

        reads = []

        def read_values(index: int) -> NDArrayNum:
            """Record one batch read and return four known absolute differences."""

            reads.append(index)
            return np.column_stack((np.zeros(4), np.arange(index * 4 + 1, index * 4 + 5, dtype=float)))

        blocks = [da.from_delayed(dask.delayed(read_values)(index), shape=(4, 2), dtype=float) for index in range(3)]
        source = xr.Dataset(
            {"distance": ("pair", np.ones(12)), "value": (("pair", "endpoint"), da.concatenate(blocks))}
        )
        eager_values = np.column_stack((np.zeros(12), np.arange(1, 13.0)))
        eager = xr.Dataset({"distance": ("pair", np.ones(12)), "value": (("pair", "endpoint"), eager_values)})
        options: dict[str, Any] = {"estimator": "matheron", "bins": bins}
        expected = gu.Variogram.from_pairs(eager, **options)

        # Run worker calls locally to record reads
        assert isinstance(source.value.data, da.Array)
        assert not reads
        with dask.config.set(scheduler="synchronous"):
            result = gu.Variogram.from_pairs(source, **options, mp_config=MultiprocConfig(chunks=4))

        # Check one read per block and equality
        assert sorted(reads) == [0, 1, 2]
        assert isinstance(source.value.data, da.Array)
        assert isinstance(result.semivariance, np.ndarray)
        np.testing.assert_array_equal(result.counts, expected.counts)
        np.testing.assert_array_equal(result.lags, expected.lags)
        np.testing.assert_array_equal(result.semivariance, expected.semivariance)


class TestVariogramErrors:
    """Test module for errors/warnings raised by variogram estimation and conversion."""

    def test_to_dict__error_custom_variogram(self) -> None:
        """Checks an error is raised when JSON export cannot preserve a custom variogram function."""

        model = VariogramModel(
            "stretched_variogram", effective_range=8, partial_sill=2, function=stretched_variogram, parameters=(1.3,)
        )

        with pytest.raises(ValueError, match="pickle instead of JSON"):
            model.to_dict()

    @pytest.mark.parametrize("min_count", [0, -1, 1.5, True, "100"])
    def test_fit__error_invalid_min_count(self, min_count: Any) -> None:
        """Checks an error is raised for a minimum pair count that is not a positive integer."""

        empirical = gu.Variogram(lags=np.arange(3.0), semivariance=np.arange(3.0), counts=np.full(3, 100))

        with pytest.raises(ValueError, match="min_count must be a positive integer"):
            empirical.fit(min_count=min_count)

    def test_fit__error_insufficient_pairs(self) -> None:
        """Checks an error is raised when fewer than two finite bins meet the minimum pair count."""

        empirical = gu.Variogram(lags=np.arange(3.0), semivariance=np.arange(3.0), counts=np.array([99, 100, 99]))

        with pytest.raises(ValueError, match="At least two finite empirical lag classes with at least 100 pairs"):
            empirical.fit(min_count=100)

    @pytest.mark.parametrize(
        "kind,model,message",
        [
            ("unknown", None, "Argument 'kind'"),
            ("correlation", None, "positive sill"),
            ("correlation", VariogramModel("spherical", effective_range=1, partial_sill=0), "positive sill"),
        ],
    )
    def test_plot__error_invalid_correlation(self, kind: Any, model: VariogramModel | None, message: str) -> None:
        """Checks an error is raised for an unknown plot kind or correlation without a positive fitted sill."""

        pytest.importorskip("matplotlib")
        variogram = gu.Variogram(lags=np.array([1.0]), semivariance=np.array([0.5]), counts=np.array([10]), model=model)

        with pytest.raises(ValueError, match=message):
            variogram.plot(kind=kind)

    @pytest.mark.parametrize(
        "options,error_type,message",
        [
            ({"estimator": "percentile", "percentile": value}, ValueError, "finite and between 0 and 100")
            for value in (-1, 101, np.nan, np.inf)
        ]
        + [
            ({"estimator": "minmax", "percentile": 90}, ValueError, "requires estimator='percentile'"),
            ({"method": "unknown"}, ValueError, "Argument method"),
            ({"reduction_backend": "flox", "method": "unknown"}, ValueError, "Argument method"),
            ({"reduction_backend": "flox", "tdigest_compression": 5}, ValueError, "tdigest_compression"),
            ({"reduction_backend": "flox", "tdigest_compression": np.inf}, ValueError, "tdigest_compression"),
            ({"flox_engine": "numba"}, ValueError, "flox_engine requires"),
            (
                {"estimator": "matheron", "reduction_backend": "flox", "mp_config": MultiprocConfig(chunks=3)},
                ValueError,
                "Flox backend does not support multiprocessing",
            ),
        ]
        + [
            ({"estimator": estimator}, TypeError, "estimator must be a SciKit-GStat estimator name or a callable")
            for estimator in (None, 1, ["matheron"])
        ]
        + [
            (
                {"estimator": estimator, "reduction_backend": backend},
                ValueError,
                f"Unknown SciKit-GStat estimator {estimator!r}",
            )
            for backend in ("geoutils", "flox")
            for estimator in ("unknown", "np", "__dict__")
        ],
    )
    def test_variogram__error_reduction_options(
        self, options: dict[str, Any], error_type: type[Exception], message: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks an error is raised for invalid reduction options, before sampling occurs."""

        # Create raster and pair data, then prevent sampling
        raster = gu.Raster.from_array(np.ones((3, 3)), from_origin(0, 3, 1, 1), 32633)
        pairs = xr.Dataset(
            {"distance": ("pair", [1.0, 2.0]), "value": (("pair", "endpoint"), [[0.0, 1.0], [0.0, 2.0]])}
        )

        def fail_sampling(*args: Any, **kwargs: Any) -> None:
            """Fail if invalid reduction options reach pair sampling."""

            pytest.fail("Invalid reduction options must fail before pair sampling")

        monkeypatch.setattr(type(raster), "pairsample", fail_sampling)

        # Check the error for both APIs
        with pytest.raises(error_type, match=message):
            gu.Variogram.from_pairs(pairs, **options)
        with pytest.raises(error_type, match=message):
            raster.variogram(**options)

    @pytest.mark.parametrize("from_pairs", [False, True])
    @pytest.mark.parametrize("estimator", ["genton", "entropy", np.mean])
    def test_methods__error_unsupported_flox(
        self, estimator: Any, from_pairs: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks an error is raised for unsupported Flox estimators."""

        # Create raster and pair data
        pairs = xr.Dataset(
            {"distance": ("pair", [1.0, 2.0]), "value": (("pair", "endpoint"), [[0.0, 1.0], [0.0, 2.0]])}
        )
        raster = gu.Raster.from_array(np.ones((3, 3)), from_origin(0, 3, 1, 1), 32633)

        def fail_sampling(*args: Any, **kwargs: Any) -> None:
            """Fail if backend validation reaches pair selection."""

            pytest.fail("Unsupported Flox requests must fail before pair sampling")

        monkeypatch.setattr(type(raster), "pairsample", fail_sampling)

        # Check the error for both APIs
        with pytest.raises(ValueError, match="Flox variogram backend supports") as error:
            if from_pairs:
                gu.Variogram.from_pairs(pairs, estimator=estimator, reduction_backend="flox")
            else:
                raster.variogram(estimator=estimator, reduction_backend="flox")

        # Check the suggested alternatives
        name = repr(estimator) if isinstance(estimator, str) else "a custom callable"
        assert f"Estimator {name} has no registered chunk reduction" in str(error.value)
        assert "method='groupwise'" in str(error.value)
        assert "reduction_backend='geoutils'" in str(error.value)

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
            gu.Variogram.from_pairs(pairs, estimator="matheron")

    @pytest.mark.parametrize("n_runs", [0, -1, 1.5, True])
    def test_variogram__error_invalid_repetitions(self, n_runs: int | float) -> None:
        """Checks that invalid repetition counts are rejected before sampling."""

        # Create a small valid raster because this option should fail before pair sampling
        raster = gu.Raster.from_array(np.arange(16, dtype=float).reshape(4, 4), from_origin(0, 4, 1, 1), 32633)

        # Reject zero, negative, fractional, and boolean run counts
        with pytest.raises(ValueError, match="Argument ``n_runs`` must be a positive integer"):
            raster.variogram(n_runs=n_runs, estimator="matheron")

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

    @pytest.mark.skipif(
        find_spec("flox") is not None or find_spec("crick") is not None,
        reason="Only runs if flox and crick are missing.",
    )
    def test_from_pairs__error_missing_dependency(self) -> None:
        """Checks an error is raised for missing optional reduction packages."""

        # Create valid pairs for both optional reductions
        pairs = xr.Dataset(
            {"distance": ("pair", [1.0, 2.0]), "value": (("pair", "endpoint"), [[0.0, 1.0], [0.0, 2.0]])}
        )

        # Check missing packages are named in the errors
        with pytest.raises(ImportError, match="flox"):
            gu.Variogram.from_pairs(pairs, estimator="matheron", bins=[1, 3], reduction_backend="flox")
        with pytest.raises(ImportError, match="crick"):
            gu.Variogram.from_pairs(pairs, estimator="dowd", bins=[1, 3], reduction_backend="flox", method="map-reduce")

        # Check native reduction works without either package
        native = gu.Variogram.from_pairs(pairs, estimator="matheron", bins=[1, 3])
        np.testing.assert_array_equal(native.counts, [2])

    @pytest.mark.parametrize("backend", ["geoutils", "flox"])
    @pytest.mark.parametrize("estimator", ["unknown", "np", "__dict__"])
    def test_from_pairs__error_unknown_estimator(self, backend: Literal["geoutils", "flox"], estimator: str) -> None:
        """Checks an error is raised for unknown estimators before reading pairs."""

        # Create pair data with uneven chunks
        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        eager = xr.Dataset(
            {"distance": ("pair", np.arange(1, 6, dtype=float)), "value": (("pair", "endpoint"), np.ones((5, 2)))}
        )
        lazy = eager.chunk({"pair": 3})
        options: dict[str, Any] = {"estimator": estimator, "reduction_backend": backend}

        def fail_read(*args: Any) -> None:
            """Fail if an invalid estimator schedules any pair data before raising its error."""

            pytest.fail("Unknown estimators must fail before reading pair data")

        # Check errors before reading pairs
        with pytest.raises(ValueError, match="Unknown SciKit-GStat estimator") as eager_error:
            gu.Variogram.from_pairs(eager, **options)
        with Callback(start=fail_read), pytest.raises(ValueError) as lazy_error:
            gu.Variogram.from_pairs(lazy, **options)

        # Check error messages and loading
        assert str(lazy_error.value) == str(eager_error.value)
        choices = "Choose from 'matheron', 'cressie', 'dowd', 'genton', 'minmax', 'entropy', 'percentile'"
        assert choices in str(lazy_error.value)
        assert isinstance(lazy.distance.data, da.Array)
        assert isinstance(lazy.value.data, da.Array)

    @pytest.mark.parametrize("backend", ["geoutils", "flox"])
    @pytest.mark.parametrize("case", ["nonpositive", "nonfinite", "empty"])
    def test_from_pairs__error_no_chunked_pairs(self, backend: Literal["geoutils", "flox"], case: str) -> None:
        """Checks an error is raised when no valid lazy pairs remain."""

        # Create lazy pairs with invalid or empty data
        da = pytest.importorskip("dask.array")

        if backend == "flox":
            pytest.importorskip("flox")
        distances = np.array([0, -1, 0, -2, 0], dtype=np.float32)
        values = np.ones((5, 2), dtype=np.float32)
        if case == "nonfinite":
            distances[:] = 1
            values[:, 1] = np.nan
        elif case == "empty":
            distances, values = distances[:0], values[:0]
        pairs = xr.Dataset(
            {
                "distance": ("pair", da.from_array(distances, chunks=3)),
                "value": (("pair", "endpoint"), da.from_array(values, chunks=(3, 2))),
            }
        )

        # Check the error and input laziness
        with pytest.raises(ValueError, match="no finite observations with positive distance"):
            gu.Variogram.from_pairs(pairs, estimator="matheron", reduction_backend=backend, bins=[1, 2, 4])
        assert isinstance(pairs.distance.data, da.Array)
        assert isinstance(pairs.value.data, da.Array)
