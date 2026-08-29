"""
Large data tests for every Dask and multiprocessing operation.

Each operation and backend (Dask, multiprocessing) runs in a separate process with an input raster whose
size exceeds the memory limit. The operation must complete with less additional worker memory than the full raster
would require.

A test fails if the input opens eagerly, the operation raises an error or times out, its representative result or
output file is incorrect, a Dask worker is replaced, multiprocessing workers disappear, or additional worker memory
reaches the size of the full raster.

This is a pass/fail computation and memory test (not a benchmark like the ASV suite).

Default Pytest run skips this module. Run it with ``python -m pytest --large-data -m large_data -ra``.
The parameters can be changed by defining environment variables before the above call (defaults shown below):
``GEOUTILS_LARGE_DATA_SHAPE`` (12288), ``GEOUTILS_LARGE_DATA_CHUNKS`` (1024),
``GEOUTILS_LARGE_DATA_MEMORY_LIMIT`` (512MB), ``GEOUTILS_LARGE_DATA_PROFILE_INTERVAL`` (0.1 seconds) and
``GEOUTILS_LARGE_DATA_TIMEOUT`` (1800 seconds).
Shape and chunk variables accept one square size or ``rows,columns``.

Pytest records failed tests in .pytest_cache/v/cache/lastfailed.
During local development, add ``--lf`` to the large data command to rerun only those cases and save time!
"""

from __future__ import annotations

import multiprocessing
import os
import traceback
from dataclasses import replace
from multiprocessing.connection import Connection

import numpy as np
import pytest

from benchmarks.workflows.registry import (
    OPERATION_BENCHMARK_CASES,
    OPERATION_BY_NAME,
    split_operation_case,
)
from benchmarks.workflows.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    logical_raster_size_mb,
    memory_limit_mb,
)
from geoutils.interface.gridding import GriddingMethod

pytestmark = [pytest.mark.large_data, pytest.mark.memory, pytest.mark.allow_logging_warnings]


def _shape_from_env(name: str, default: tuple[int, int]) -> tuple[int, int]:
    """Read raster shape from environment variable."""

    # Keep the documented default when no override is requested
    value = os.environ.get(name)
    if value is None:
        return default

    # Accept separate row and column sizes for non-square test rasters
    if "," in value:
        rows, cols = value.split(",", maxsplit=1)
        return int(rows), int(cols)

    # A single value is a convenient square-raster shorthand
    size = int(value)
    return size, size


@pytest.fixture(scope="module")
def large_data_config(tmp_path_factory: pytest.TempPathFactory) -> BenchmarkConfig:
    """Create one larger-than-memory fixture configuration shared by all operations."""

    # Reusing sources avoids writing the same large raster for every backend case
    directory = tmp_path_factory.mktemp("geoutils-large-data")
    return BenchmarkConfig(
        shape=_shape_from_env("GEOUTILS_LARGE_DATA_SHAPE", (12288, 12288)),
        # Larger chunks retain 144 tasks while keeping the canonical raster above memory
        chunks=_shape_from_env("GEOUTILS_LARGE_DATA_CHUNKS", (1024, 1024)),
        memory_limit=os.environ.get("GEOUTILS_LARGE_DATA_MEMORY_LIMIT", "512MB"),
        profile_interval=float(os.environ.get("GEOUTILS_LARGE_DATA_PROFILE_INTERVAL", "0.1")),
        subsample_size=2048,
        ninterp=2048,
        point_partition_size=8,
        # Release native workspaces between bounded writes for the strict memory contract
        trim_dask_memory=True,
        directory=str(directory),
    )


def _execute_operation(case_name: str, config: BenchmarkConfig) -> BenchmarkResult:
    """Run one case in a fresh process and retain its laziness assertions and metrics."""

    # A fresh process prevents one backend's imports and allocators from changing the next baseline
    backend, operation = split_operation_case(case_name)
    with BenchmarkRunner(backend, config) as runner:
        raster = runner.make_raster()

        # Inputs must point to chunks or their source file before any operation graph is built
        if backend == "dask":
            assert raster.data.chunks is not None
            assert not raster._in_memory
        else:
            assert not raster.is_loaded

        # One execution provides correctness, worker health and memory metrics
        result = runner.run(operation)

        # Input objects must still be lazy or file-backed after the operation completes
        if backend == "dask":
            assert not raster._in_memory
        else:
            assert not raster.is_loaded
    return result


def _operation_process(sender: Connection, case_name: str, config: BenchmarkConfig) -> None:
    """Send one isolated result or a complete traceback back to Pytest."""

    try:
        # Direct child processes may start their own Dask or multiprocessing workers
        result = _execute_operation(case_name, config)
    except BaseException:  # noqa: B036
        sender.send((False, traceback.format_exc()))
    else:
        sender.send((True, result))
    finally:
        sender.close()


def _run_isolated(case_name: str, config: BenchmarkConfig) -> BenchmarkResult:
    """Execute one large data case in a clean spawned process with a bounded timeout."""

    # Spawn avoids inheriting native caches retained by a previous backend operation
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_operation_process, args=(sender, case_name, config))
    process.start()
    sender.close()

    # The timeout turns a dead worker tree into one actionable Pytest failure
    timeout = float(os.environ.get("GEOUTILS_LARGE_DATA_TIMEOUT", "1800"))
    if not receiver.poll(timeout):
        process.terminate()
        process.join()
        pytest.fail(f"Large data case exceeded {timeout:g} seconds: {case_name}")

    try:
        succeeded, payload = receiver.recv()
    except EOFError:
        process.join()
        pytest.fail(f"Large data process exited without a result: {case_name} ({process.exitcode})")
    finally:
        receiver.close()
    process.join()

    # Tracebacks from the isolated process remain visible in the parent test report
    if not succeeded:
        pytest.fail(str(payload), pytrace=False)
    if process.exitcode != 0:
        pytest.fail(f"Large data process exited with status {process.exitcode}: {case_name}")
    return payload  # type: ignore[return-value]


class TestLargeData:
    """Check every registered out-of-core operation against one explicit memory contract."""

    def _check_case(
        self,
        case_name: str,
        large_data_config: BenchmarkConfig,
    ) -> None:
        """Run one operation and check its result, worker health and bounded memory."""

        # Every backend needs psutil and Dask additionally needs distributed workers
        pytest.importorskip("psutil")
        backend, operation = split_operation_case(case_name)
        if backend == "dask":
            pytest.importorskip("dask")
            pytest.importorskip("distributed")
            if operation == "grid":
                pytest.importorskip("dask_geopandas")

        # The uncompressed input must exceed the configured limit before claiming a large data test
        logical_mb = logical_raster_size_mb(large_data_config)
        configured_limit_mb = memory_limit_mb(large_data_config.memory_limit)
        assert logical_mb > configured_limit_mb

        # Per-case process isolation makes memory independent of preceding parametrizations
        result = _run_isolated(case_name, large_data_config)

        # Validate the small fingerprint and any large file produced by the operation
        expected_value = OPERATION_BY_NAME[operation].expected_value
        assert np.isclose(result.value, expected_value, equal_nan=True)
        if result.output_file is not None:
            assert os.path.exists(result.output_file)
            # Remove each checked output before the next large operation starts
            os.remove(result.output_file)

        if backend == "dask":
            # Dask must complete without hiding a failure through nanny replacement
            assert not result.worker_restarted

            # Native imports form a directly measured baseline before the operation starts
            baseline_worker_mb = result.dask_worker_baseline_mem_mb
            assert baseline_worker_mb is not None
            peak_worker_mb = result.metrics.peak_dask_worker_process_mem_mb
            assert peak_worker_mb is not None

            # Additional worker memory must remain below the complete logical raster
            assert peak_worker_mb - baseline_worker_mb < logical_mb
        else:
            # Multiprocessing normally recycles workers to release native-library caches
            assert result.worker_pids_before
            assert result.worker_pids_after

            # Native imports form a fixed baseline that is unrelated to raster loading
            child_trace = result.metrics.child_process_mem_mb
            assert child_trace
            baseline_child_mb = child_trace[0][1]
            peak_child_mb = result.metrics.peak_child_process_mem_mb
            assert peak_child_mb is not None

            # Additional worker memory must remain below the complete logical raster
            assert peak_child_mb - baseline_child_mb < logical_mb

            # Aggregate process-tree memory is retained for reports and diagnostics
            assert result.metrics.peak_process_tree_mem_mb is not None

    @pytest.mark.parametrize("case_name", OPERATION_BENCHMARK_CASES)
    def test_operation_stays_out_of_core(self, case_name: str, large_data_config: BenchmarkConfig) -> None:
        """Complete one larger-than-memory operation without loading its full raster."""

        self._check_case(case_name=case_name, large_data_config=large_data_config)

    @pytest.mark.parametrize("case_name", ["dask-grid", "multiprocessing-grid"])
    @pytest.mark.parametrize("resampling", ["idw", "mean"])
    def test_neighborhood_gridding_stays_out_of_core(
        self,
        case_name: str,
        resampling: GriddingMethod,
        large_data_config: BenchmarkConfig,
    ) -> None:
        """Complete the IDW and circular-statistic execution paths through every out-of-core backend."""

        # A two-pixel support crosses chunk edges without creating unbounded point-cell pairs
        config = replace(
            large_data_config,
            operation_method=resampling,
            grid_dist_nodata_pixel=2,
        )

        # IDW has its own reduction, while mean represents the shared circular-statistic neighborhood path
        self._check_case(case_name=case_name, large_data_config=config)
