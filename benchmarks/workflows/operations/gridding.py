"""Define point cloud gridding benchmarks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np

from benchmarks.workflows.config import (
    DEFAULT_CHUNK_SIZE,
    GRID_POINT_COUNTS,
    POINT_CHUNK_SIZE,
    RASTER_SIZES,
    WORKER_RASTER_SIZES,
    ExecutionMode,
    Parameter,
    RuntimeConfig,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    merge_cases,
    parameter_config,
    reference_case,
)

ORDER = 90
METHODS = ("nearest", "linear", "idw", "mean")
POINTS_PER_AXIS = {method: 17 for method in METHODS}
METHOD_LABELS = {
    "nearest": "Nearest",
    "linear": "Linear (Delaunay)",
    "idw": "Inverse-distance",
    "mean": "Circular mean",
}
ENGINE_LABELS = {"scipy": "SciPy", "numba": "Numba"}

######################################
# Define setup for gridding operation
######################################


def write_point_source(filename: str, points_per_axis: int = 5) -> None:
    """Write a regular constant point cloud for gridding scenarios."""

    if os.path.exists(filename):
        return
    if points_per_axis < 1:
        raise ValueError("Points per axis must be strictly positive")

    # Keep points away from the exact border so every geometry is unambiguous
    coords_x = np.linspace(7.05, 7.95, points_per_axis)
    coords_y = np.linspace(45.05, 45.95, points_per_axis)
    xx, yy = np.meshgrid(coords_x, coords_y)
    points = gpd.GeoDataFrame(
        {"z": np.ones(xx.size, dtype=np.float64)},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=4326,
    )
    points.to_file(filename, driver="GPKG")


def grid_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Build the public gridding options used for execution and labels."""

    return {
        "shape": config.shape,
        "bounds": (7.0, 45.0, 8.0, 46.0),
        "resampling": case.method,
        "dist_nodata_pixel": config.value("grid_dist_nodata_pixel", float("inf")),
        "engine": case.engine,
        "chunksizes": config.chunks,
        # One SciPy thread keeps backend and GDAL comparisons repeatable
        "n_threads": 1,
    }


def prepare_grid(runner: Any, case: Case) -> None:
    """Write the regular point cloud shared by GeoUtils and GDAL gridding."""

    write_point_source(
        runner.path("source-points.gpkg"),
        int(runner.config.value("point_features_per_axis", 5)),
    )


def run_grid(runner: Any, case: Case) -> float:
    """Grid the prepared point cloud and complete its raster output."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)

    import geoutils as gu

    # Point input and raster output both remain partitioned for their backend
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
    pointcloud = (
        gu.open_pointcloud(
            runner.path("source-points.gpkg"),
            data_column="z",
            chunks=runner.config.value("point_partition_size", POINT_CHUNK_SIZE),
        )
        if runner.backend == "dask"
        else gu.PointCloud(runner.path("source-points.gpkg"), data_column="z")
    )
    options = grid_options(case, runner.config)
    output = (
        pointcloud.pc.grid(**options) if runner.backend == "dask" else pointcloud.grid(**options, mp_config=mp_config)
    )
    return runner._compute_raster(output)


GRID = Operation(
    "grid",
    prepare_grid,
    run_grid,
    grid_options,
    ("resampling", "engine"),
    label="Gridding",
    order=15,
    large_data_cases=tuple(
        Case(method="nearest", engine="scipy", execution=execution, options={"point_partition_size": 8})
        for execution in ("dask", "multiprocessing")
    ),
    large_data_dependencies=("dask_geopandas",),
)
OPERATIONS = (GRID,)


###################
# Define benchmarks
###################


# Each case fixes one method, engine and execution mode for one ASV result series; its sweep owns the changing input
def benchmark_case(
    method: str,
    engine: str,
    execution: ExecutionMode,
    *,
    variant: str | None = None,
) -> Case:
    """Build one gridding case while keeping its declaration compact."""

    return Case(
        method=method,
        engine=engine,
        execution=execution,
        labels={"method": METHOD_LABELS[method], "engine": ENGINE_LABELS[engine]},
        variant=variant,
    )


# Compare all four gridding methods across execution modes while keeping SciPy as the calculation engine
MODE_CASES = {
    method: tuple(
        benchmark_case(
            method,
            "scipy",
            execution,
        )
        for execution in ("inmem", "dask", "multiprocessing")
    )
    for method in METHODS
}

# Reuse the in-memory SciPy cases in one plot that isolates the choice of gridding method
METHOD_CASES = tuple(MODE_CASES[method][0] for method in METHODS)

# Compare SciPy and Numba in memory for the methods supported by both calculation engines
ENGINE_CASES = {
    method: (
        MODE_CASES[method][0],
        benchmark_case(method, "numba", "inmem"),
    )
    for method in ("nearest", "idw", "mean")
}

# Repeat the nearest engine comparison while varying source point count instead of raster size
POINT_ENGINE_CASES = tuple(benchmark_case("nearest", engine, "inmem") for engine in ("scipy", "numba"))

# Add one fixed-size run per Numba method and worker execution mode to check that compiled kernels work there
# The in-memory engine comparisons already measure how these methods scale with raster size
WORKER_CASES = tuple(
    benchmark_case(
        method,
        "numba",
        execution,
        variant="worker",
    )
    for method in ("nearest", "idw", "mean")
    for execution in ("dask", "multiprocessing")
)
REFERENCES = {method: reference_case(MODE_CASES[method], implementation="gdal") for method in METHODS}
POINT_REFERENCE = reference_case(POINT_ENGINE_CASES, implementation="gdal")


def raster_size(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Place the selected raster size around the common source point input."""

    # Keep the point count fixed so only the method and its required support distance differ
    method_distances = {"nearest": float("inf"), "linear": float("inf"), "idw": 16.0, "mean": 16.0}
    if case.method not in method_distances:
        raise ValueError(f"No gridding fixture is defined for method {case.method!r}")
    assert parameter is not None
    size = int(parameter)
    values = {
        "shape": (size, size),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "point_features_per_axis": POINTS_PER_AXIS[case.method],
        "grid_dist_nodata_pixel": method_distances[case.method],
    }
    if case.execution == "dask":
        values["point_partition_size"] = POINT_CHUNK_SIZE
    return values


def point_count(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Place the selected point count in an otherwise fixed configuration."""

    assert parameter is not None
    return {"point_features_per_axis": int(parameter)}


def grid_workload(parameter: Parameter, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the output raster, chunks and source point grid."""

    config = configs[0]
    points = int(config.value("point_features_per_axis", 5))
    parts = [
        f"{config.shape[0]:,} × {config.shape[1]:,} raster",
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks",
        f"{points:,} × {points:,} source points",
    ]
    if any(item.value("point_partition_size") is not None for item in configs):
        parts.append(f"{int(config.value('point_partition_size', POINT_CHUNK_SIZE)):,}-point chunks")
    return "; ".join(parts)


RASTER_CASES = merge_cases(*MODE_CASES.values(), *ENGINE_CASES.values())

# Measure the selected cases and matching GDAL references over raster size, point count or the fixed worker check
BENCHMARKS = (
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        GRID,
        (*RASTER_CASES, *REFERENCES.values()),
        raster_size,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=grid_workload,
    ),
    parameter_config(
        "points_per_axis",
        GRID_POINT_COUNTS,
        GRID,
        (*POINT_ENGINE_CASES, POINT_REFERENCE),
        point_count,
        parameter_label="Number of source points per axis",
        parameter_title="source point count",
        describe_workload=grid_workload,
    ),
    parameter_config(
        "raster_size",
        WORKER_RASTER_SIZES,
        GRID,
        WORKER_CASES,
        raster_size,
        name="grid-worker",
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=grid_workload,
    ),
)


#############################
# Define report comparisons
#############################

# Comparisons select saved series for method, engine and execution plots, adding matching GDAL series where available
COMPARISONS = (
    *tuple(
        comparison(
            BENCHMARKS[0],
            by="execution",
            cases=(*MODE_CASES[method], REFERENCES[method]),
            slug="gridding-raster-size" if method == "nearest" else f"{method}-gridding-raster-size",
            documentation=method == "nearest",
        )
        for method in METHODS
    ),
    comparison(
        BENCHMARKS[0],
        by="method",
        cases=METHOD_CASES,
        slug="gridding-method-raster-size",
        documentation=False,
    ),
    *tuple(
        comparison(
            BENCHMARKS[0],
            by="engine",
            cases=(*ENGINE_CASES[method], REFERENCES[method]),
            slug=f"{method}-gridding-engine-raster-size",
            documentation=False,
        )
        for method in ("nearest", "idw", "mean")
    ),
    comparison(
        BENCHMARKS[1],
        by="engine",
        slug="nearest-gridding-engine-point-count",
        documentation=False,
    ),
)
