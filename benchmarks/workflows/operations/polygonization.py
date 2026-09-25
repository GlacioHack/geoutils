"""Define polygonization benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geopandas as gpd

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
    raster_size_config,
    strategy_cases,
)

ORDER = 60
STRATEGIES = ("label_union", "label_stitch", "geometry_stitch")
POLYGON_OPTIONS = {"polygon_regions_per_axis": 21}


def polygonize_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public polygonization options used for execution and labels."""

    options: dict[str, Any] = {"target_values": 1}
    if case.strategy is not None:
        options["strategy"] = case.strategy
    return options


def run_polygonize(runner: Any, case: BenchmarkCase) -> float:
    """Polygonize the prepared regions and write the complete vector output."""

    raster = runner.make_raster(runner.polygon_raster_file)

    # The selected chunk strategy reconciles polygons that cross output tiles
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    options = polygonize_options(case, runner.config)
    polygons = (
        raster.rst.polygonize(**options)
        if runner.backend == "dask"
        else raster.polygonize(**options, mp_config=mp_config)
    )
    runner._last_output_file = runner._output_path(case.operation, suffix=".gpkg")
    polygon_data = polygons if isinstance(polygons, gpd.GeoDataFrame) else polygons.ds
    polygon_data.to_file(runner._last_output_file)
    return float(len(polygon_data))


OPERATIONS = (
    Operation(
        "polygonize",
        run_polygonize,
        polygonize_options,
        ("strategy",),
        {None: ("rasterio",)},
        strategies=STRATEGIES,
        default_strategy="label_stitch",
        coverage=OperationCoverage(11),
    ),
)


def strategy_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Keep smaller chunks so every strategy crosses many block boundaries."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (500, 500)}


########################################
# Cases, sweeps and report comparisons
########################################


# Each case fixes one execution mode or strategy for one ASV result series; the sweep owns the changing raster size
EXECUTION_CASES = execution_cases(
    "polygonize",
    None,
    "rasterio",
    strategy="label_stitch",
    options=POLYGON_OPTIONS,
)
STRATEGY_CASES = strategy_cases(
    "polygonize",
    None,
    "rasterio",
    STRATEGIES,
    execution="dask",
    options=POLYGON_OPTIONS,
    variant="strategy",
)
REFERENCE = external_case(EXECUTION_CASES)

# Measure the execution cases with their GDAL reference and the strategy cases over the same raster-size axis
SWEEPS = (
    Sweep(RASTER_AXIS, raster_size_config, EXECUTION_CASES, (REFERENCE,)),
    Sweep(RASTER_AXIS, strategy_size, STRATEGY_CASES, name="polygonize-strategy"),
)

# Comparisons select saved series for report plots: GeoUtils with GDAL, then the GeoUtils strategies by themselves
COMPARISONS = (
    comparison(SWEEPS[0]),
    comparison(SWEEPS[1], documentation=False),
)
