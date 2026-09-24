"""Define raster filtering benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    RASTER_CHUNK_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
)

ORDER = 50


def filter_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public filter options used for execution and labels."""

    return {"method": case.method, "engine": case.engine, "size": 5}


def run_filter(runner: Any, case: BenchmarkCase) -> float:
    """Apply the selected local filter and complete its output."""

    raster = runner.make_raster()

    # Apply a local operation before writing its complete large output
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    options = filter_options(case, runner.config)
    output = raster.rst.filter(**options) if runner.backend == "dask" else raster.filter(**options, mp_config=mp_config)
    return runner._compute_raster(output, case.operation)


OPERATIONS = (
    Operation(
        "filter",
        run_filter,
        filter_options,
        ("method", "engine"),
        {"mean": ("scipy",)},
        "mean",
    ),
)
COVERAGE = (OperationCoverage("filter", ("dask", "multiprocessing"), 1, 4),)


def _chunk_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the selected square chunk size."""

    size = int(parameter)
    return {"chunks": (size, size)}


_CASES = execution_cases("filter-chunk-size", "filter", "mean", "scipy", executions=("dask", "multiprocessing"))
SWEEPS = (
    Sweep(
        "chunk_size",
        RASTER_CHUNK_AXIS,
        _chunk_size,
        _CASES,
        base={"shape": (2_000, 2_000)},
    ),
)
COMPARISONS = (comparison(SWEEPS[0]),)
