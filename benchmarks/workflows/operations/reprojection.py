"""Define raster reprojection benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    RASTER_SIZES,
    RuntimeConfig,
    raster_size_config,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
    reference_case,
)
from benchmarks.workflows.io import write_constant_raster

ORDER = 40

#########################################
# Define setup for reprojection operation
#########################################


def reproject_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Define reprojection options."""

    return {
        "crs": 32632,
        "grid_size": config.shape[::-1],
        "resampling": case.method,
        "nodata": -99999,
        "n_threads": 1,
        "memory_limit": 64,
    }


def prepare_reproject(runner: Any, case: Case) -> None:
    """Prepare the raster to be reprojected."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)


def run_reproject(runner: Any, case: Case) -> float:
    """Run reprojection and ensure output computes (Dask/MP)."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)

    raster = runner.make_raster()
    options = reproject_options(case, runner.config)

    # Fix the target size so GeoUtils and GDAL references write the same pixel count
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    output = (
        raster.rst.reproject(**options)
        if runner.backend == "dask"
        else raster.reproject(**options, mp_config=mp_config)
    )
    return runner._compute_raster(output)


REPROJECT = Operation(
    "reproject",
    prepare_reproject,
    run_reproject,
    reproject_options,
    ("resampling",),
    label="Reprojection",
    order=5,
    large_data_cases=tuple(
        Case(method="nearest", engine="rasterio", execution=execution) for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (REPROJECT,)


#####################################
# Define benchmarks and comparisons
#####################################


CASES = execution_cases(
    "nearest",
    "rasterio",
    labels={"method": "Nearest", "engine": "Rasterio/GDAL"},
)
REFERENCE = reference_case(CASES, implementation="gdal")
BENCHMARKS = (
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        REPROJECT,
        (*CASES, REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    ),
)
COMPARISONS = (comparison(BENCHMARKS[0], by="execution"),)
