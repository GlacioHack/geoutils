"""Define benchmarks for basic raster operations and clipping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    RASTER_SIZES,
    VECTOR_COMPARISON_OPTIONS,
    RuntimeConfig,
    raster_size_config,
)
from benchmarks.workflows.core import (
    Benchmark,
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
    reference_case,
)
from benchmarks.workflows.io import write_constant_raster, write_vector_source

ORDER = 30

############################################
# Define setup for basic raster operations
############################################


def raster_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Return the public options used by one basic raster operation."""

    return {
        "crop": {"bbox": (7.1, 45.1, 7.9, 45.9)},
        "clip": {},
        "translate": {"xoff": 0.1, "yoff": 0.1},
        "copy": {"deep": False},
        "write": {},
    }[config.value("operation")]


def prepare_raster_operation(runner: Any, case: Case) -> None:
    """Create the raster and any clipping vector needed by one basic operation."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)
    if runner.operation.name == "clip":
        write_vector_source(
            runner.path("source-vector.gpkg"),
            int(runner.config.value("vector_features_per_axis", 1)),
        )


def run_raster_operation(runner: Any, case: Case) -> float:
    """Run one basic raster operation and complete its output."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)
    if case.implementation != "geoutils":
        raise ValueError(f"Unsupported {runner.operation.name} implementation: {case.implementation}")

    operation = runner.operation.name
    raster = runner.make_raster()
    options = raster_options(case, runner.config)

    if operation == "crop":
        if runner.backend != "dask":
            raise ValueError("Deferred raster cropping is only registered for Dask")

        # Crop metadata and array indexes lazily before writing the selected region
        output = raster.rst.crop(**options)
    elif operation == "clip":
        from geoutils import Vector

        # Mask the constant raster with the same regular polygons passed to the GDAL cutline reference
        clipping_vector = Vector(runner.path("source-vector.gpkg"))
        mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
        output = (
            raster.rst.clip(clipping_vector)
            if runner.backend == "dask"
            else raster.clip(clipping_vector, mp_config=mp_config)
        )
    elif operation == "translate":
        if runner.backend != "dask":
            raise ValueError("Deferred raster translation is only registered for Dask")

        # Translation changes georeferencing while leaving every value chunk deferred
        output = raster.rst.translate(**options)
    elif operation == "copy":
        if runner.backend != "dask":
            raise ValueError("Lazy raster copying is only registered for Dask")

        # A shallow accessor copy duplicates metadata without evaluating the graph
        output = raster.rst.copy(**options)
    else:
        if runner.backend != "dask":
            raise ValueError("Direct lazy writing is only registered for Dask")

        # Write the unchanged lazy source to isolate the storage path
        output = raster

    return runner._compute_raster(output)


# List each operation tested out of core, its supported execution modes and its expected result value
# Both the fixed ASV benchmarks and large-data tests use this coverage list
CROP = Operation(
    "crop",
    prepare_raster_operation,
    run_raster_operation,
    raster_options,
    order=0,
    large_data_cases=(Case(execution="dask", options={"operation": "crop"}),),
)
CLIP = Operation(
    "clip",
    prepare_raster_operation,
    run_raster_operation,
    raster_options,
    label="Clipping",
    order=1,
    large_data_cases=tuple(
        Case(execution=execution, options={"operation": "clip"}) for execution in ("dask", "multiprocessing")
    ),
)
TRANSLATE = Operation(
    "translate",
    prepare_raster_operation,
    run_raster_operation,
    raster_options,
    order=2,
    large_data_cases=(Case(execution="dask", options={"operation": "translate"}),),
)
COPY = Operation(
    "copy",
    prepare_raster_operation,
    run_raster_operation,
    raster_options,
    ("deep",),
    order=3,
    large_data_cases=(Case(execution="dask", options={"operation": "copy"}),),
)
WRITE = Operation(
    "write",
    prepare_raster_operation,
    run_raster_operation,
    raster_options,
    order=12,
    large_data_cases=(Case(execution="dask", options={"operation": "write"}),),
)
OPERATIONS = (CROP, CLIP, TRANSLATE, COPY, WRITE)


#####################################
# Define benchmarks and comparisons
#####################################


# Each case fixes one GeoUtils execution mode for one ASV result series; the sweep owns the changing raster size
# The external case identifies the matching GDAL series, which the default comparison plots with the GeoUtils series
CLIP_CASES = execution_cases(
    None,
    None,
    options={**VECTOR_COMPARISON_OPTIONS, "operation": "clip"},
)
CLIP_REFERENCE = reference_case(CLIP_CASES, implementation="gdal")


def clip_workload(parameter: int | float, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the raster, chunks and clipping polygon grid."""

    config = configs[0]
    features = int(config.value("vector_features_per_axis", 1))
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} raster; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {features:,} × {features:,} vector features"
    )


BENCHMARKS = (
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        CLIP,
        (*CLIP_CASES, CLIP_REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=clip_workload,
    ),
)
COMPARISONS = (comparison(BENCHMARKS[0], by="execution"),)
