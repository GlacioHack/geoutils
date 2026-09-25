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
    Sweep,
    execution_cases,
    external_case,
    raster_size_config,
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
        coverage=OperationCoverage(5),
    ),
)


########################################
# Cases, sweeps and report comparisons
########################################


# Each case fixes one GeoUtils execution mode for one ASV result series; the sweep owns the changing raster size
# The external case identifies the matching GDAL series, which the default comparison plots with the GeoUtils series
CASES = execution_cases("reproject", "nearest", "rasterio")
REFERENCE = external_case(CASES)
SWEEPS = (Sweep(RASTER_AXIS, raster_size_config, CASES, (REFERENCE,)),)
