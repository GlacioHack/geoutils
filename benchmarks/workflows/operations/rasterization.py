"""Define vector rasterization and mask benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.workflows.config import (
    RASTER_AXIS,
    VECTOR_COMPARISON_OPTIONS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Sweep,
    execution_cases,
    external_case,
    raster_size_config,
)

ORDER = 70


def rasterize_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public rasterization options shared by both execution paths."""

    options: dict[str, Any] = {
        "shape": config.shape,
        "bounds": (7.0, 45.0, 8.0, 46.0),
        "crs": 4326,
        "chunksizes": config.chunks,
    }
    if case.operation == "rasterize":
        options.update({"in_value": 1, "out_value": 0, "out_dtype": np.uint8})
    return options


def run_rasterize(runner: Any, case: BenchmarkCase) -> float:
    """Rasterize the prepared polygons or create their boolean mask."""

    from geoutils import Vector

    # Vector input is small while the produced raster is larger than memory
    vector = Vector(runner.vector_file)
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    options = rasterize_options(case, runner.config)
    output = getattr(vector, case.operation)(
        dask=runner.backend == "dask",
        mp_config=mp_config,
        **options,
    )
    return runner._compute_raster(output, case.operation)


OPERATIONS = (
    Operation(
        "rasterize",
        run_rasterize,
        rasterize_options,
        method_engines={None: ("rasterio",)},
        coverage=OperationCoverage(13),
    ),
    Operation("create_mask", run_rasterize, rasterize_options, coverage=OperationCoverage(14)),
)


########################################
# Cases, sweeps and report comparisons
########################################


# Each case fixes one GeoUtils execution mode for one ASV result series; the sweep owns the changing raster size
# The external case identifies the matching GDAL series, which the default comparison plots with the GeoUtils series
CASES = execution_cases(
    "rasterize",
    None,
    "rasterio",
    options=VECTOR_COMPARISON_OPTIONS,
)
REFERENCE = external_case(CASES)
SWEEPS = (Sweep(RASTER_AXIS, raster_size_config, CASES, (REFERENCE,)),)
