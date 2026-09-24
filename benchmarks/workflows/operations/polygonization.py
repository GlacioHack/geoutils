"""Define raster polygonization benchmarks."""

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
    strategy_cases,
)

ORDER = 60
_STRATEGIES = ("label_union", "label_stitch", "geometry_stitch")


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
        strategies=_STRATEGIES,
        default_strategy="label_stitch",
    ),
)
COVERAGE = (OperationCoverage("polygonize", ("dask", "multiprocessing"), 1, 11),)


def _execution_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the scheduled raster and chunk sizes for execution comparisons."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000)}


def _strategy_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Keep smaller chunks so every strategy crosses many block boundaries."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (500, 500)}


_EXECUTION_CASES = execution_cases(
    "polygonization-raster-size", "polygonize", None, "rasterio", strategy="label_stitch"
)
_STRATEGY_CASES = strategy_cases(
    "polygonization-strategy-raster-size",
    "polygonize",
    None,
    "rasterio",
    _STRATEGIES,
    execution="dask",
)
_REFERENCE = external_case(_EXECUTION_CASES)

SWEEPS = (
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _execution_size,
        _EXECUTION_CASES,
        (_REFERENCE,),
        base={"polygon_regions_per_axis": 21},
    ),
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _strategy_size,
        _STRATEGY_CASES,
        base={"polygon_regions_per_axis": 21},
    ),
)
COMPARISONS = (
    comparison(SWEEPS[0]),
    comparison(SWEEPS[1], documentation=False),
)
