"""Prepare equivalent GeoUtils and Flox grouped-statistics benchmark runs."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from benchmarks.asv_suite import asv_pr_check_enabled
from benchmarks.flox_comparison.reference import grouped_stats as flox_grouped_stats
from benchmarks.workflows.config import (
    BenchmarkCase,
    BenchmarkConfig,
    ExecutionMode,
    ExternalReferenceCase,
    Parameter,
    Sweep,
    process_tree_memory_increase_mb,
)
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.profiler import profile_call


def prepare_grouped_inputs(
    size: int,
    groups_per_axis: int,
    layout: Literal["local", "interleaved"],
    execution_mode: ExecutionMode,
    chunks: tuple[int, int] = (1_000, 1_000),
) -> tuple[Any, Any, Any, NDArrayNum]:
    """Create equivalent eager or Dask values, groups, categories and selection for both implementations."""

    # Prepare two bounded signals with different missing observations
    positions = np.arange(size * size).reshape(size, size)
    first = (positions % 997).astype(np.float64) * 0.125
    values = np.stack((first, first * 2 + 10))
    values[0, positions % 17 == 0] = np.nan
    values[1, positions % 29 == 0] = np.nan

    # Keep group membership and the common selection independent of missing values
    rows, columns = np.arange(size)[:, None], np.arange(size)[None, :]
    if layout == "local":
        groups = (rows * groups_per_axis // size) * groups_per_axis + columns * groups_per_axis // size
    else:
        groups = (rows % groups_per_axis) * groups_per_axis + columns % groups_per_axis
    groups = groups.astype(np.int32)
    mask = positions % 13 != 0
    categories = np.arange(groups_per_axis**2)

    # Give both implementations the same lazy arrays and spatial partitions
    if execution_mode == "dask":
        import dask.array as da

        values = da.from_array(values, chunks=(1, *chunks))
        groups = da.from_array(groups, chunks=chunks)
        mask = da.from_array(mask, chunks=chunks)
    return values, groups, mask, categories


def compute_geoutils_grouped_stats(
    values: Any,
    groups: Any,
    mask: Any,
    categories: NDArrayNum,
    *,
    mp_config: MultiprocConfig | None = None,
) -> pd.DataFrame:
    """Compute the comparison statistics through public GeoUtils stats()."""

    from geoutils.stats import stats

    return stats(
        {"first": values[0], "second": values[1]},
        ("mean", "std"),
        by={"zone": groups},
        categories={"zone": categories},
        mask=mask,
        observed=False,
        mp_config=mp_config,
    )


@dataclass(frozen=True)
class FloxRunner:
    """Run the independent Flox reference on already prepared arrays."""

    inputs: tuple[Any, Any, Any, NDArrayNum]
    execution_mode: ExecutionMode

    def execute(self) -> pd.DataFrame:
        """Calculate and format the complete Flox result."""

        return flox_grouped_stats(*self.inputs, use_dask=self.execution_mode == "dask")


class _GroupedFloxBenchmark:
    """Compare complete GeoUtils and Flox results after preparing identical arrays."""

    timeout = 900
    number = 1
    repeat = 2
    rounds = 1
    warmup_time = 0
    sweep: Sweep
    case: BenchmarkCase | ExternalReferenceCase

    def make_config(self, parameter: Parameter) -> BenchmarkConfig:
        """Build one configuration from the operation-local sweep."""

        return self.sweep.make_config(parameter, self.case, asv_pr_check_enabled())

    def setup(self, parameter: Parameter) -> None:
        """Prepare common arrays and optional workers outside the measurement."""

        if asv_pr_check_enabled() and not self.case.pr_check:
            raise NotImplementedError("Benchmark case omitted from the pull-request sample")
        self.external = isinstance(self.case, ExternalReferenceCase)
        execution_mode = self.case.execution_mode
        assert execution_mode is not None
        if self.external:
            try:
                importlib.import_module("flox")
            except ImportError as exc:
                raise NotImplementedError("Install optional flox to run this comparison") from exc

        # Fix the scheduler for both implementations and construct all observations before timing starts
        self.dask = importlib.import_module("dask")
        config = self.make_config(parameter)

        # Start one persistent worker before building arrays so it receives only serialized tiles
        self.mp_cluster: MpCluster | None = None
        self.mp_config: MultiprocConfig | None = None
        if execution_mode == "multiprocessing":
            self.mp_cluster = MpCluster({"nb_workers": 1, "max_tasks_per_child": None})
            self.mp_config = MultiprocConfig(chunks=config.chunks, cluster=self.mp_cluster)

        # Keep complete prepared inputs in the client for the GeoUtils or independent Flox call
        self.inputs = prepare_grouped_inputs(
            config.shape[0],
            config.grouped_regions_per_axis,
            config.grouped_layout,
            execution_mode,
            config.chunks,
        )
        self.flox_runner = FloxRunner(self.inputs, execution_mode) if self.external else None

    def teardown(self, parameter: Parameter) -> None:
        """Stop worker processes and release arrays after each independent ASV measurement."""

        cluster = getattr(self, "mp_cluster", None)
        if cluster is not None:
            cluster.close()
        if hasattr(self, "inputs"):
            del self.inputs

    def time_operation(self, parameter: Parameter) -> None:
        """Compute every requested result with one threaded or multiprocessing worker."""

        with self.dask.config.set(scheduler="threads", num_workers=1):
            if self.flox_runner is not None:
                self.flox_runner.execute()
            else:
                compute_geoutils_grouped_stats(*self.inputs, mp_config=self.mp_config)

    def track_end_to_end_time_s(self, parameter: Parameter) -> float:
        """Measure masking, grouping and complete output construction from prepared inputs."""

        start = time.perf_counter()
        self.time_operation(parameter)
        return time.perf_counter() - start

    def track_process_tree_mem_increase_mb(self, parameter: Parameter) -> float:
        """Measure peak memory increase while the complete grouped result is calculated."""

        _, metrics = profile_call(self.time_operation, parameter, dask=False, include_children=True)
        return process_tree_memory_increase_mb(metrics)


setattr(_GroupedFloxBenchmark.track_end_to_end_time_s, "unit", "seconds")
setattr(_GroupedFloxBenchmark.track_process_tree_mem_increase_mb, "unit", "MB")
