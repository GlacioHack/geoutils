"""Define point cloud gridding benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    GRID_POINT_AXIS,
    RASTER_AXIS,
    WORKER_INTEGRATION_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    CalculationEngine,
    ExecutionMode,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    external_case,
    merge_cases,
)

ORDER = 90
_METHODS = ("nearest", "linear", "idw", "mean")
_POINTS_PER_AXIS = {method: 17 for method in _METHODS}

############################
# Operation execution
############################


def grid_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public gridding options used for execution and labels."""

    return {
        "shape": config.shape,
        "bounds": (7.0, 45.0, 8.0, 46.0),
        "resampling": case.method,
        "dist_nodata_pixel": config.grid_dist_nodata_pixel,
        "engine": case.engine,
        "chunksizes": config.chunks,
        # One SciPy thread keeps backend and GDAL comparisons repeatable
        "n_threads": 1,
    }


def run_grid(runner: Any, case: BenchmarkCase) -> float:
    """Grid the prepared point cloud and complete its raster output."""

    import geoutils as gu

    # Point input and raster output both remain partitioned for their backend
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    pointcloud = (
        gu.open_pointcloud(
            runner.point_file,
            data_column="z",
            chunks=runner.config.point_partition_size,
        )
        if runner.backend == "dask"
        else gu.PointCloud(runner.point_file, data_column="z")
    )
    options = grid_options(case, runner.config)
    output = (
        pointcloud.pc.grid(**options) if runner.backend == "dask" else pointcloud.grid(**options, mp_config=mp_config)
    )
    return runner._compute_raster(output, case.operation)


OPERATIONS = (
    Operation(
        "grid",
        run_grid,
        grid_options,
        ("resampling", "engine"),
        {
            "nearest": ("scipy", "numba"),
            "linear": ("scipy",),
            "idw": ("scipy", "numba"),
            "mean": ("scipy", "numba"),
        },
        "nearest",
    ),
)
COVERAGE = (OperationCoverage("grid", ("dask", "multiprocessing"), 1, 15),)


############################
# Cases and input sweeps
############################


def _case(
    sweep_id: str,
    method: str,
    engine: CalculationEngine,
    execution: ExecutionMode,
    *,
    pr_check: bool = False,
) -> BenchmarkCase:
    """Build one gridding case while keeping its declaration compact."""

    return BenchmarkCase(sweep_id, "grid", method, engine, execution, pr_check=pr_check)


# Compare all four gridding methods across execution modes while keeping SciPy as the calculation engine
_MODE_CASES = {
    method: tuple(
        _case(
            "gridding-raster-size",
            method,
            "scipy",
            execution,
            pr_check=method == "nearest",
        )
        for execution in ("eager", "dask", "multiprocessing")
    )
    for method in _METHODS
}

# Reuse the eager SciPy cases in one plot that isolates the choice of gridding method
_METHOD_CASES = tuple(_MODE_CASES[method][0] for method in _METHODS)

# Compare SciPy and Numba in eager mode for the methods supported by both calculation engines
_ENGINE_CASES = {
    method: (
        _MODE_CASES[method][0],
        _case("gridding-raster-size", method, "numba", "eager", pr_check=method == "nearest"),
    )
    for method in ("nearest", "idw", "mean")
}

# Repeat the nearest engine comparison while varying source point count instead of raster size
_POINT_ENGINE_CASES = tuple(_case("gridding-point-count", "nearest", engine, "eager") for engine in ("scipy", "numba"))

# Add one fixed-size run per Numba method and worker execution mode to check that compiled kernels work there
# The eager engine comparisons already measure how these methods scale with raster size
_WORKER_CASES = tuple(
    _case(
        "worker-integration",
        method,
        "numba",
        execution,
        pr_check=(method, execution) in (("idw", "dask"), ("mean", "multiprocessing")),
    )
    for method in ("nearest", "idw", "mean")
    for execution in ("dask", "multiprocessing")
)
_REFERENCES = {method: external_case(_MODE_CASES[method], pr_check=method == "nearest") for method in _METHODS}
_POINT_REFERENCE = external_case(_POINT_ENGINE_CASES)


def _raster_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Place the selected raster size around the common source point input."""

    # Keep the point count fixed so only the method and its required support distance differ
    method_distances = {"nearest": float("inf"), "linear": float("inf"), "idw": 16.0, "mean": 16.0}
    if case.method not in method_distances:
        raise ValueError(f"No gridding fixture is defined for method {case.method!r}")
    size = int(parameter)
    return {
        "shape": (size, size),
        "chunks": (1_000, 1_000),
        "point_features_per_axis": _POINTS_PER_AXIS[case.method],
        "grid_dist_nodata_pixel": method_distances[case.method],
    }


def _point_count(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Place the selected point count in an otherwise fixed configuration."""

    return {"point_features_per_axis": int(parameter)}


_RASTER_CASES = merge_cases(*_MODE_CASES.values(), *_ENGINE_CASES.values())
SWEEPS = (
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _raster_size,
        _RASTER_CASES,
        tuple(_REFERENCES.values()),
    ),
    Sweep(
        "points_per_axis",
        GRID_POINT_AXIS,
        _point_count,
        _POINT_ENGINE_CASES,
        (_POINT_REFERENCE,),
        base={"shape": (2_000, 2_000), "chunks": (1_000, 1_000), "grid_dist_nodata_pixel": float("inf")},
    ),
    Sweep("raster_size", WORKER_INTEGRATION_AXIS, _raster_size, _WORKER_CASES),
)


############################
# Report comparisons
############################

COMPARISONS = (
    *tuple(
        comparison(
            SWEEPS[0],
            cases=_MODE_CASES[method],
            references=(_REFERENCES[method],),
            slug="gridding-raster-size" if method == "nearest" else f"{method}-gridding-raster-size",
            documentation=method == "nearest",
        )
        for method in _METHODS
    ),
    comparison(
        SWEEPS[0],
        cases=_METHOD_CASES,
        references=(),
        slug="gridding-method-raster-size",
        documentation=False,
    ),
    *tuple(
        comparison(
            SWEEPS[0],
            cases=_ENGINE_CASES[method],
            references=(_REFERENCES[method],),
            slug=f"{method}-gridding-engine-raster-size",
            documentation=False,
        )
        for method in ("nearest", "idw", "mean")
    ),
    comparison(
        SWEEPS[1],
        slug="nearest-gridding-engine-point-count",
        documentation=False,
    ),
)
