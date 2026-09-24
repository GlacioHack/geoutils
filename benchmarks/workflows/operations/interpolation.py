"""Define raster point-interpolation benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.workflows.config import (
    POINT_COUNT_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
)

ORDER = 20


def interpolation_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public interpolation options used for execution and labels."""

    return {"method": case.method, "as_array": True}


def run_interpolation(runner: Any, case: BenchmarkCase) -> float:
    """Interpolate the prepared raster at deterministic point coordinates."""

    raster = runner.make_raster()

    # A uniform distribution touches many chunks and avoids incomplete edge support
    rng = np.random.default_rng(42)
    points = (
        rng.uniform(7.01, 7.99, size=runner.config.ninterp),
        rng.uniform(45.01, 45.99, size=runner.config.ninterp),
    )
    options = interpolation_options(case, runner.config)
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    values = (
        raster.rst.interp_points(points, **options)
        if runner.backend == "dask"
        else raster.interp_points(points, **options, mp_config=mp_config)
    )
    if hasattr(values, "compute"):
        values = values.compute()
    return float(np.nanmean(values))


OPERATIONS = (
    Operation(
        "interp_points",
        run_interpolation,
        interpolation_options,
        ("method",),
        {"linear": ("scipy",)},
        "linear",
    ),
)
COVERAGE = (OperationCoverage("interp_points", ("dask", "multiprocessing"), 1, 10),)


def _point_count(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the number of interpolation coordinates."""

    return {"ninterp": int(parameter)}


_CASES = execution_cases("interpolation-point-count", "interp_points", "linear", "scipy")
SWEEPS = (
    Sweep(
        "interpolated_points",
        POINT_COUNT_AXIS,
        _point_count,
        _CASES,
        base={"shape": (2_000, 2_000), "chunks": (1_000, 1_000)},
    ),
)
COMPARISONS = (comparison(SWEEPS[0], logarithmic_x=True),)
