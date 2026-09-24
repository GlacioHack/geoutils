"""Define raster reprojection benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    RASTER_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
    external_case,
)

ORDER = 40


def reproject_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public reprojection options used for execution and labels."""

    return {
        "crs": 32632,
        "grid_size": config.shape[::-1],
        "resampling": case.method,
        "nodata": -99999,
        "n_threads": 1,
        "memory_limit": 64,
    }


def run_reproject(runner: Any, case: BenchmarkCase) -> float:
    """Reproject the prepared raster and complete its output."""

    raster = runner.make_raster()
    options = reproject_options(case, runner.config)

    # Fix the target size so GeoUtils and GDAL references write the same pixel count
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    output = (
        raster.rst.reproject(**options)
        if runner.backend == "dask"
        else raster.reproject(**options, mp_config=mp_config)
    )
    return runner._compute_raster(output, case.operation)


OPERATIONS = (
    Operation(
        "reproject",
        run_reproject,
        reproject_options,
        ("resampling",),
        {"nearest": ("rasterio",)},
        "nearest",
    ),
)
COVERAGE = (OperationCoverage("reproject", ("dask", "multiprocessing"), 1, 5),)


def _raster_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the selected square raster around the scheduled chunk size."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000)}


_CASES = execution_cases("reprojection-raster-size", "reproject", "nearest", "rasterio")
_REFERENCE = external_case(_CASES)
SWEEPS = (
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _raster_size,
        _CASES,
        (_REFERENCE,),
    ),
)
COMPARISONS = (comparison(SWEEPS[0]),)
