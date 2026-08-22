"""Large-data Dask memory stress tests."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from benchmarks._dask_workflows import (
    DaskRasterWorkflowConfig,
    DaskRasterWorkflowRunner,
)

pytestmark = [pytest.mark.large_data, pytest.mark.memory, pytest.mark.allow_logging_warnings]


def _shape_from_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = os.environ.get(name)
    if value is None:
        return default
    if "," in value:
        rows, cols = value.split(",", maxsplit=1)
        return int(rows), int(cols)
    size = int(value)
    return size, size


def _stress_config(tmp_path: Any, *, shape: tuple[int, int] | None = None, memory_limit: str | None = None) -> Any:
    shape = shape or _shape_from_env("GEOUTILS_DASK_STRESS_SHAPE", (12288, 12288))
    return DaskRasterWorkflowConfig(
        shape=shape,
        chunks=_shape_from_env("GEOUTILS_DASK_STRESS_CHUNKS", (1024, 1024)),
        memory_limit=memory_limit or os.environ.get("GEOUTILS_DASK_STRESS_MEMORY_LIMIT", "512MB"),
        spill_directory=str(tmp_path),
        profile_interval=float(os.environ.get("GEOUTILS_DASK_STRESS_PROFILE_INTERVAL", "0.1")),
        subsample_size=2048,
        ninterp=2048,
    )


def _logical_size_mb(config: DaskRasterWorkflowConfig) -> float:
    return config.shape[0] * config.shape[1] * np.dtype("float32").itemsize / 1_000_000


def _memory_limit_mb(memory_limit: str) -> float:
    memory_limit = memory_limit.strip().lower()
    if memory_limit.endswith("gb"):
        return float(memory_limit[:-2]) * 1000
    if memory_limit.endswith("gib"):
        return float(memory_limit[:-3]) * 1024
    if memory_limit.endswith("mb"):
        return float(memory_limit[:-2])
    if memory_limit.endswith("mib"):
        return float(memory_limit[:-3]) * 1024 / 1000
    return float(memory_limit) / 1_000_000


def test_large_subsample_completes_without_worker_restart(tmp_path: Any) -> None:
    """A larger-than-memory lazy raster should complete a chunked GeoUtils subsample/reduction."""

    pytest.importorskip("dask")
    pytest.importorskip("distributed")

    config = _stress_config(tmp_path)
    assert _logical_size_mb(config) > _memory_limit_mb(config.memory_limit)

    with DaskRasterWorkflowRunner(config) as runner:
        raster = runner.make_raster()
        assert raster.data.chunks is not None
        assert not raster._in_memory

        result = runner.run("subsample_topk")

    assert np.isclose(result.value, config.raster_value)
    assert not result.worker_restarted
    assert result.metrics.peak_dask_worker_process_rss_mb is not None


def test_peak_worker_memory_scales_sublinearly_with_total_raster_size(tmp_path: Any) -> None:
    """Increasing total raster size with fixed chunks should not scale peak worker RSS proportionally."""

    pytest.importorskip("dask")
    pytest.importorskip("distributed")

    memory_limit = os.environ.get("GEOUTILS_DASK_STRESS_MEMORY_LIMIT", "512MB")
    small_config = _stress_config(tmp_path / "small", shape=(4096, 4096), memory_limit=memory_limit)
    large_config = _stress_config(tmp_path / "large", shape=(8192, 8192), memory_limit=memory_limit)

    with DaskRasterWorkflowRunner(small_config) as runner:
        small_result = runner.run("subsample_topk")

    with DaskRasterWorkflowRunner(large_config) as runner:
        large_result = runner.run("subsample_topk")

    assert np.isclose(small_result.value, small_config.raster_value)
    assert np.isclose(large_result.value, large_config.raster_value)
    assert not small_result.worker_restarted
    assert not large_result.worker_restarted

    small_peak = small_result.metrics.peak_dask_worker_process_rss_mb
    large_peak = large_result.metrics.peak_dask_worker_process_rss_mb
    assert small_peak is not None
    assert large_peak is not None

    logical_ratio = _logical_size_mb(large_config) / _logical_size_mb(small_config)
    peak_ratio = large_peak / small_peak
    assert peak_ratio < logical_ratio


def test_large_persist_spills_without_worker_restart(tmp_path: Any) -> None:
    """Persisting a larger-than-memory lazy raster should spill instead of restarting workers."""

    pytest.importorskip("dask")
    pytest.importorskip("distributed")

    config = _stress_config(tmp_path, shape=(12288, 12288), memory_limit="512MB")
    config.profile_interval = 0.05
    assert _logical_size_mb(config) > _memory_limit_mb(config.memory_limit)

    with DaskRasterWorkflowRunner(config) as runner:
        result = runner.run("persist_reduce")

    assert np.isclose(result.value, 2 * config.raster_value)
    assert not result.worker_restarted
    assert result.metrics.peak_dask_spilled_mb is not None
    assert result.metrics.peak_dask_spilled_mb > 0
