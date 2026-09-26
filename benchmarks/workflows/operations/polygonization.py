"""Define polygonization benchmarks."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio as rio

from benchmarks.workflows.config import (
    RASTER_SIZES,
    Parameter,
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
    strategy_cases,
)
from benchmarks.workflows.io import tiff_block_size

ORDER = 60
STRATEGIES = ("label_union", "label_stitch", "geometry_stitch")
STRATEGY_LABELS = {
    "label_union": "Label union",
    "label_stitch": "Label stitch",
    "geometry_stitch": "Geometry stitch",
}
POLYGON_OPTIONS = {"polygon_regions_per_axis": 21}

###########################################
# Define setup for polygonization operation
###########################################


def write_polygon_raster(filename: str, config: RuntimeConfig) -> None:
    """Write regularly spaced connected regions for polygonization scenarios."""

    if os.path.exists(filename):
        return
    regions = int(config.value("polygon_regions_per_axis", 1))
    if regions < 1:
        raise ValueError("Polygon regions per axis must be strictly positive")

    # Separate rectangles with value of 1 with nodata so every rectangle is one region
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = tiff_block_size(height, config.chunks[0])
    block_x = tiff_block_size(width, config.chunks[1])

    # Write per block
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=4326,
        transform=transform,
        nodata=-99999,
        tiled=True,
        blockxsize=block_x,
        blockysize=block_y,
        compress="DEFLATE",
        BIGTIFF="IF_NEEDED",
    ) as destination:
        for _, window in destination.block_windows(1):
            # Pixel phases locate the inner half of every regular grid cell
            row_start = int(window.row_off)
            col_start = int(window.col_off)
            rows = np.arange(row_start, row_start + int(window.height))
            cols = np.arange(col_start, col_start + int(window.width))
            row_phase = ((rows + 0.5) * regions / height) % 1
            col_phase = ((cols + 0.5) * regions / width) % 1
            inside_rows = (row_phase >= 0.25) & (row_phase <= 0.75)
            inside_cols = (col_phase >= 0.25) & (col_phase <= 0.75)
            inside = inside_rows[:, None] & inside_cols[None, :]

            # Nodata gaps keep neighboring rectangles disconnected for both engines
            block = np.full(inside.shape, -99999, dtype=np.float32)
            block[inside] = config.raster_value
            destination.write(block, indexes=1, window=window)


def polygonize_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Build the public polygonization options used for execution and labels."""

    options: dict[str, Any] = {"target_values": 1}
    if case.strategy is not None:
        options["strategy"] = case.strategy
    return options


def prepare_polygonize(runner: Any, case: Case) -> None:
    """Write regularly spaced regions for GeoUtils and GDAL polygonization."""

    write_polygon_raster(runner.path("source-polygonize.tif"), runner.config)


def run_polygonize(runner: Any, case: Case) -> float:
    """Polygonize the prepared regions and write the complete vector output."""

    if case.implementation == "gdal":
        from benchmarks.comparisons.gdal import execute_gdal

        return execute_gdal(runner, case)

    raster = runner.make_raster(runner.path("source-polygonize.tif"))

    # The selected chunk strategy reconciles polygons that cross output tiles
    mp_config = runner._multiproc_config(suffix=".gpkg") if runner.backend == "multiprocessing" else None
    options = polygonize_options(case, runner.config)
    polygons = (
        raster.rst.polygonize(**options)
        if runner.backend == "dask"
        else raster.polygonize(**options, mp_config=mp_config)
    )
    runner._last_output_file = runner._output_path(suffix=".gpkg")
    polygon_data = polygons if isinstance(polygons, gpd.GeoDataFrame) else polygons.ds
    polygon_data.to_file(runner._last_output_file)
    return float(len(polygon_data))


POLYGONIZE = Operation(
    "polygonize",
    prepare_polygonize,
    run_polygonize,
    polygonize_options,
    ("strategy",),
    label="Polygonization",
    order=11,
    large_data_cases=tuple(
        Case(engine="rasterio", execution=execution, strategy="label_stitch")
        for execution in ("dask", "multiprocessing")
    ),
)
OPERATIONS = (POLYGONIZE,)


def strategy_size(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Keep smaller chunks so every strategy crosses many block boundaries."""

    assert parameter is not None
    size = int(parameter)
    return {"shape": (size, size), "chunks": (500, 500)}


#####################################
# Define benchmarks and comparisons
#####################################


# Each case fixes one execution mode or strategy for one ASV result series; the sweep owns the changing raster size
EXECUTION_CASES = execution_cases(
    None,
    "rasterio",
    strategy="label_stitch",
    options=POLYGON_OPTIONS,
    labels={"engine": "Rasterio/GDAL", "strategy": STRATEGY_LABELS["label_stitch"]},
)
STRATEGY_CASES = strategy_cases(
    None,
    "rasterio",
    STRATEGIES,
    execution="dask",
    options=POLYGON_OPTIONS,
    strategy_labels=STRATEGY_LABELS,
    variant="strategy",
)
REFERENCE = reference_case(EXECUTION_CASES, implementation="gdal")


def polygon_workload(parameter: Parameter, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the raster, chunks and regular regions used by polygonization."""

    config = configs[0]
    regions = int(config.value("polygon_regions_per_axis", 1))
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} raster; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {regions:,} × {regions:,} raster regions"
    )


# Measure the execution cases with their GDAL reference and the strategy cases over the same raster-size axis
BENCHMARKS = (
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        POLYGONIZE,
        (*EXECUTION_CASES, REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=polygon_workload,
    ),
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        POLYGONIZE,
        STRATEGY_CASES,
        strategy_size,
        name="polygonize-strategy",
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=polygon_workload,
    ),
)

# Comparisons select saved series for report plots: GeoUtils with GDAL, then the GeoUtils strategies by themselves
COMPARISONS = (
    comparison(BENCHMARKS[0], by="execution"),
    comparison(BENCHMARKS[1], by="strategy", documentation=False),
)
