"""Tests for lightweight variogram statistics, fitting and conversions."""

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


@pytest.fixture(autouse=True)
def _writable_matplotlib_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep optional plotting imports inside the test workspace."""

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path))


def test_importing_geoutils_does_not_load_variogram_backends() -> None:
    """Optional fitting and covariance packages should stay unloaded during an ordinary import."""

    code = """
import sys
import geoutils
assert not {'skgstat', 'gstools', 'gpytorch', 'torch'}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_variogram_is_small_immutable_and_serializable() -> None:
    """The public holder should own only immutable lag arrays and plain metadata."""

    result = gu.Variogram(
        lags=np.array([1.0, 2.0]),
        semivariance=np.array([0.2, 0.5]),
        counts=np.array([10, 8]),
        semivariance_error=np.array([0.02, 0.03]),
        bin_lower_edges=np.array([0.5, 1.5]),
        bin_edges=np.array([1.5, 2.5]),
        fitted_semivariance=np.array([0.25, 0.45]),
        model=gu.VariogramModel("gaussian", effective_range=4, partial_sill=0.8, nugget=0.1),
        estimator="matheron",
    )

    with pytest.raises(ValueError, match="read-only"):
        result.lags[0] = 3
    restored = gu.Variogram.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.model == result.model
    assert restored.semivariance_error is not None and result.semivariance_error is not None
    assert np.array_equal(restored.semivariance_error, result.semivariance_error)
    assert set(restored.to_xarray().data_vars) == {
        "semivariance",
        "semivariance_error",
        "count",
        "bin_lower_edge",
        "bin_edge",
        "fitted_semivariance",
    }


def test_from_pairs_does_not_retain_pair_data() -> None:
    """Empirical reduction should keep bin metadata but release arrays for individual pairs."""

    values = np.column_stack((np.arange(20, dtype=float), np.arange(20, dtype=float) + 2))
    pairs = xr.Dataset(
        {"value": (("pair", "endpoint"), values), "distance": ("pair", np.linspace(1, 20, 20))},
        coords={"pair": np.arange(20), "endpoint": ["first", "second"]},
    )

    result = gu.Variogram.from_pairs(pairs, estimator="matheron", bins="log", n_lags=4)

    assert len(result.lags) == 4
    assert np.sum(result.counts) == 20
    assert result.backend_object is None
    assert result.semivariance == pytest.approx(np.full(4, 2.0))


def test_from_pairs_uses_sampled_distance_limits_for_stable_bins() -> None:
    """Pair metadata should define reusable bins even when sampled endpoints miss the exact limits."""

    pairs = xr.Dataset(
        {
            "value": (("pair", "endpoint"), np.column_stack((np.zeros(6), np.arange(1, 7)))),
            "distance": ("pair", np.linspace(2, 8, 6)),
        },
        attrs={"min_distance": 1.0, "max_distance": 10.0},
    )

    result = gu.Variogram.from_pairs(pairs, bins="log", n_lags=3)

    assert result.bin_lower_edges is not None and result.bin_edges is not None
    assert result.bin_lower_edges[0] == 1
    assert result.bin_edges[-1] == 10
    assert np.sum(result.counts) == 6


def test_object_variogram_aggregates_runs_and_fits_summed_model() -> None:
    """The easy interface should estimate sampling error and fit without retaining pairs."""

    y, x = np.mgrid[:35, :35]
    raster = gu.Raster.from_array(np.sin(x / 4) + 0.5 * np.cos(y / 10), from_origin(0, 35, 2, 2), 32633, nodata=None)
    result = raster.variogram(
        n_pairs=1_000,
        n_lags=8,
        n_runs=3,
        model=["gaussian", "spherical"],
        random_state=42,
    )

    assert result.model is not None and result.model.model_name == "sum"
    assert [component.model_name for component in result.model.components] == ["gaussian", "spherical"]
    assert np.any(np.isfinite(result.semivariance_error))
    assert result.backend_object is None
    assert result.attrs["n_runs"] == 3


def test_fit_accepts_short_names_and_skgstat_model_functions() -> None:
    """The lightweight fit should retain xDEM's accepted model name forms."""

    skgstat = pytest.importorskip("skgstat")
    lags = np.linspace(1, 20, 12)
    empirical = gu.Variogram(
        lags=lags,
        semivariance=skgstat.models.gaussian(lags, 10, 2),
        counts=np.full(12, 100),
    )

    fitted = empirical.fit([skgstat.models.gaussian, "Sph"])

    assert fitted.model is not None
    assert [component.model_name for component in fitted.model.components] == ["gaussian", "spherical"]


def test_pointcloud_variogram_and_advanced_pairs_share_api() -> None:
    """Point cloud users should be able to inspect pairs or request only lag statistics."""

    y, x = np.mgrid[:18, :18]
    pointcloud = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (np.sin(x / 3) + np.cos(y / 5)).ravel(), crs=32633)

    pairs = pointcloud.sample_pairs(n_pairs=300, min_distance=1, max_distance=15, random_state=2)
    result = pointcloud.variogram(n_pairs=300, min_lag=1, max_lag=15, n_lags=6, random_state=2)

    assert pairs.sizes == {"pair": 300, "endpoint": 2}
    assert len(result.lags) == 6
    assert result.model is None


def test_model_evaluation_and_gpytorch_parameters() -> None:
    """Portable model metadata should evaluate and describe an exact GP conversion."""

    result = gu.Variogram.from_model("gaussian", effective_range=12, partial_sill=3, nugget=0.2, active_dims=(0, 1))
    parameters = result.gpytorch_parameters()

    assert parameters["kernel_name"] == "RBF"
    assert parameters["lengthscale"] == pytest.approx(12 / (2 * np.sqrt(2)))
    assert result.variogram(0) == 0
    assert result.correlation(0) == 1
    assert result.correlation(np.zeros((2, 3))).shape == (2, 3)


def test_product_model_multiplies_covariances() -> None:
    """A product model should multiply component covariance rather than semivariance."""

    spatial = gu.Variogram.from_model("gaussian", effective_range=10, partial_sill=2, active_dims=(0, 1))
    temporal = gu.Variogram.from_model("exponential", effective_range=3, partial_sill=4, active_dims=(2,))
    combined = gu.Variogram.combine(spatial, temporal, combination="product", nugget=0.5)

    assert combined.model is not None
    assert combined.model.sill == 8.5
    assert combined.covariance(0) == pytest.approx(8.5)
    assert combined.variogram(0) == 0
    parameters = combined.gpytorch_parameters()
    assert [component["active_dims"] for component in parameters["components"]] == [(0, 1), (2,)]
    assert parameters["noise"] == 0.5


def test_skgstat_estimation_can_discard_or_retain_backend() -> None:
    """The direct coordinate adapter should retain its backend only when requested."""

    coordinates = np.linspace(0, 10, 40)[:, np.newaxis]
    values = np.sin(coordinates[:, 0])
    result = gu.Variogram.estimate(coordinates, values, model="gaussian", n_lags=6, normalize=False)
    retained = gu.Variogram.estimate(
        coordinates, values, model="gaussian", n_lags=6, normalize=False, keep_backend=True
    )

    assert result.backend_object is None
    assert retained.backend_object is not None
    assert retained.without_backend().backend_object is None


@pytest.mark.parametrize("model_name,smoothness", [("gaussian", None), ("exponential", None), ("matern", 1.5)])
def test_gstools_conversion_matches_skgstat(model_name: str, smoothness: float | None) -> None:
    """Converted models should preserve SciKit-GStat semivariances and nugget meaning."""

    gstools = pytest.importorskip("gstools")
    skgstat = pytest.importorskip("skgstat")
    result = gu.Variogram.from_model(model_name, effective_range=8, partial_sill=2, nugget=0.1, smoothness=smoothness)
    converted = result.to_gstools(dim=1)
    lags = np.array([0.0, 0.25, 1.0, 4.0, 8.0])
    model_function = getattr(skgstat.models, model_name)
    expected = (
        model_function(lags, r=8, c0=2, b=0.1)
        if smoothness is None
        else model_function(lags, r=8, c0=2, s=smoothness, b=0.1)
    )

    assert isinstance(converted.model, gstools.CovModel)
    assert converted.model.vario_axis(lags) == pytest.approx(expected)
    assert converted.active_dims is None


def test_gstools_conversion_preserves_sum_and_common_active_dims() -> None:
    """Summed structures should share one GSTools coordinate selection and one parent nugget."""

    first = gu.Variogram.from_model("gaussian", 8, 2, active_dims=(0, 1))
    second = gu.Variogram.from_model("exponential", 20, 3, active_dims=(0, 1))
    converted = gu.Variogram.combine(first, second, nugget=0.1).to_gstools(dim=2)

    assert converted.active_dims == (0, 1)
    assert converted.model.var == pytest.approx(5)
    assert converted.model.nugget == pytest.approx(0.1)


def test_gstools_rejects_component_specific_dimensions() -> None:
    """GSTools should fail clearly when a composed model requires feature selection per component."""

    spatial = gu.Variogram.from_model("gaussian", 8, 2, active_dims=(0, 1))
    temporal = gu.Variogram.from_model("exponential", 3, 1, active_dims=(2,))
    combined = gu.Variogram.combine(spatial, temporal)

    with pytest.raises(NotImplementedError, match="different dimensions"):
        combined.to_gstools(dim=3)
