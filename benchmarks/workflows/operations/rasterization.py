"""Define vector rasterization and mask benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

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
    Operation("rasterize", run_rasterize, rasterize_options, method_engines={None: ("rasterio",)}),
    Operation("create_mask", run_rasterize, rasterize_options),
)
COVERAGE = (
    OperationCoverage("rasterize", ("dask", "multiprocessing"), 1, 13),
    OperationCoverage("create_mask", ("dask", "multiprocessing"), 1, 14),
)


def _raster_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the output raster and chunk sizes around a fixed vector input."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000)}


_CASES = execution_cases("rasterization-raster-size", "rasterize", None, "rasterio")
_REFERENCE = external_case(_CASES)
SWEEPS = (
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _raster_size,
        _CASES,
        (_REFERENCE,),
        base={"vector_features_per_axis": 51, "dask_write_batch_size": 4},
    ),
)
COMPARISONS = (comparison(SWEEPS[0]),)
