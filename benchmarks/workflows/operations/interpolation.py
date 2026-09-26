"""Define raster point-interpolation benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.workflows.config import (
    DEFAULT_POINT_COUNT,
    INTERPOLATED_POINT_COUNTS,
    Parameter,
    RuntimeConfig,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
)
from benchmarks.workflows.io import write_constant_raster

ORDER = 20

##########################################
# Define setup for interpolation operation
##########################################


def interpolation_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Build the public interpolation options used for execution and labels."""

    return {"method": case.method, "as_array": True}


def prepare_interpolation(runner: Any, case: Case) -> None:
    """Write the raster sampled by every interpolation case."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)


def run_interpolation(runner: Any, case: Case) -> float:
    """Interpolate the prepared raster at deterministic point coordinates."""

    raster = runner.make_raster()

    # A uniform distribution touches many chunks and avoids incomplete edge support
    rng = np.random.default_rng(42)
    points = (
        rng.uniform(7.01, 7.99, size=runner.config.value("ninterp", DEFAULT_POINT_COUNT)),
        rng.uniform(45.01, 45.99, size=runner.config.value("ninterp", DEFAULT_POINT_COUNT)),
    )
    options = interpolation_options(case, runner.config)
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    values = (
        raster.rst.interp_points(points, **options)
        if runner.backend == "dask"
        else raster.interp_points(points, **options, mp_config=mp_config)
    )
    if hasattr(values, "compute"):
        values = values.compute()
    return float(np.nanmean(values))


INTERPOLATION = Operation(
    "interp_points",
    prepare_interpolation,
    run_interpolation,
    interpolation_options,
    ("method",),
    label="Point interpolation",
    order=10,
    large_data_cases=tuple(
        Case(method="linear", engine="scipy", execution=execution, options={"ninterp": DEFAULT_POINT_COUNT})
        for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (INTERPOLATION,)


def point_count(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Set the number of interpolation coordinates."""

    assert parameter is not None
    return {"ninterp": int(parameter)}


def interpolation_workload(parameter: Parameter, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the raster and requested interpolation points."""

    config = configs[0]
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} raster; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {int(parameter):,} interpolated points"
    )


#####################################
# Define benchmarks and comparisons
#####################################


# Each case fixes one execution mode for one ASV result series; the sweep owns the changing point count
CASES = execution_cases(
    "linear",
    "scipy",
    labels={"method": "Linear (Delaunay)", "engine": "SciPy"},
)
BENCHMARKS = (
    parameter_config(
        "interpolated_points",
        INTERPOLATED_POINT_COUNTS,
        INTERPOLATION,
        CASES,
        point_count,
        parameter_label="Number of interpolated points",
        parameter_title="point count",
        describe_workload=interpolation_workload,
    ),
)

# The comparison selects the saved execution-mode series for one report plot; it runs no additional benchmark
COMPARISONS = (comparison(BENCHMARKS[0], by="execution", logarithmic_x=True),)
