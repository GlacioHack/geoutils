"""Define benchmarks for basic raster operations and clipping."""

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
from benchmarks.workflows.fixtures import read_raster_center

ORDER = 30


def raster_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Return the public options used by one basic raster operation."""

    return {
        "crop": {"bbox": (7.1, 45.1, 7.9, 45.9)},
        "clip": {},
        "translate": {"xoff": 0.1, "yoff": 0.1},
        "copy": {"deep": False},
        "write": {},
    }[case.operation]


def run_raster_operation(runner: Any, case: BenchmarkCase) -> float:
    """Run one basic raster operation and complete its output."""

    raster = runner.make_raster()
    options = raster_options(case, runner.config)

    if case.operation == "crop":
        if runner.backend != "dask":
            raise ValueError("Deferred raster cropping is only registered for Dask")

        # Crop metadata and array indexes lazily before writing the selected region
        output = raster.rst.crop(**options)
    elif case.operation == "clip":
        from geoutils import Vector

        # Mask the constant raster with the same regular polygons passed to the GDAL cutline reference
        clipping_vector = Vector(runner.vector_file)
        mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
        output = (
            raster.rst.clip(clipping_vector)
            if runner.backend == "dask"
            else raster.clip(clipping_vector, mp_config=mp_config)
        )
    elif case.operation == "translate":
        if runner.backend != "dask":
            raise ValueError("Deferred raster translation is only registered for Dask")

        # Translation changes georeferencing while leaving every value chunk deferred
        output = raster.rst.translate(**options)
    elif case.operation == "copy":
        if runner.backend != "dask":
            raise ValueError("Lazy raster copying is only registered for Dask")

        # A shallow accessor copy duplicates metadata without evaluating the graph
        output = raster.rst.copy(**options)
    else:
        if runner.backend != "dask":
            raise ValueError("Direct lazy writing is only registered for Dask")

        # Write the unchanged lazy source to isolate the storage path
        runner._last_output_file = runner._write_dask_raster(raster, case.operation)
        return read_raster_center(runner._last_output_file)

    return runner._compute_raster(output, case.operation)


OPERATIONS = tuple(
    Operation(name, run_raster_operation, raster_options, ("deep",) if name == "copy" else ())
    for name in ("crop", "clip", "translate", "copy", "write")
)


# List each operation tested out of core, its supported execution modes and its expected result value
# Both the fixed ASV benchmarks and large-data tests use this coverage list
COVERAGE = (
    OperationCoverage("crop", ("dask",), 1, 0),
    OperationCoverage("clip", ("dask", "multiprocessing"), 1, 1),
    OperationCoverage("translate", ("dask",), 1, 2),
    OperationCoverage("copy", ("dask",), 1, 3),
    OperationCoverage("write", ("dask",), 1, 12),
)


def _clip_config(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the raster and chunk sizes used by the clipping comparison."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000)}


_CLIP_CASES = execution_cases(
    "clip-raster-size",
    "clip",
    None,
    None,
    pr_executions=("eager", "dask", "multiprocessing"),
)
_CLIP_REFERENCE = external_case(_CLIP_CASES, pr_check=True)

SWEEPS = (
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _clip_config,
        _CLIP_CASES,
        (_CLIP_REFERENCE,),
        base={"vector_features_per_axis": 51, "dask_write_batch_size": 4},
    ),
)

COMPARISONS = (comparison(SWEEPS[0]),)
