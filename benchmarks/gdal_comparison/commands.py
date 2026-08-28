"""Build GDAL commands equivalent to selected GeoUtils operations."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Literal

from benchmarks.workflows.runner import BenchmarkConfig

ComparisonOperation = Literal["reproject", "polygonize", "rasterize", "grid"]
COMPARISON_OPERATIONS: tuple[ComparisonOperation, ...] = ("reproject", "polygonize", "rasterize", "grid")
GdalGridAlgorithm = Literal[
    "nearest",
    "linear",
    "invdist",
    "average",
    "minimum",
    "maximum",
    "range",
    "count",
    "average_distance",
    "average_distance_pts",
]


@dataclass(frozen=True)
class GdalCommand:
    """Store one GDAL command and the file it must create."""

    command: list[str]
    output_file: str


def _require_command(name: str) -> str:
    """Return an installed GDAL executable or raise a clear environment error."""

    # Resolve executables once so subprocess never depends on shell parsing
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required GDAL command is not installed: {name}")
    return executable


def build_gdal_grid_command(
    point_file: str,
    output_file: str,
    *,
    algorithm: GdalGridAlgorithm,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
    radius: tuple[float, float],
    layer: str = "source-points",
    zfield: str = "z",
    min_points: int = 1,
    distance_power: float = 2.0,
    nodata: float = -9999,
    gdal_cachemax_mb: int | None = None,
    output_crs: str | None = None,
    creation_options: list[str] | None = None,
) -> GdalCommand:
    """Build one GDAL gridding command for a defined raster grid."""

    # Linear interpolation has one fallback radius while other methods use an ellipse
    if algorithm == "linear":
        if radius[0] != radius[1]:
            raise ValueError("GDAL linear gridding requires an equal X and Y fallback radius")
        algorithm_options = [f"radius={radius[0]}", f"nodata={nodata}"]
    else:
        algorithm_options = [f"radius1={radius[0]}", f"radius2={radius[1]}", "angle=0"]

        # GDAL gives inverse-distance weighting an additional power parameter
        if algorithm == "invdist":
            algorithm_options.insert(0, f"power={distance_power}")
            algorithm_options.append("smoothing=0")
            algorithm_options.append("max_points=0")
        if algorithm != "nearest":
            algorithm_options.append(f"min_points={min_points}")
        algorithm_options.append(f"nodata={nodata}")
    algorithm_definition = ":".join((algorithm, *algorithm_options))

    # Bounds describe raster edges while GeoUtils grid coordinates describe cell centers
    left, bottom, right, top = bounds
    height, width = shape
    command = [
        _require_command("gdal_grid"),
        # Fix internal parallelism so every comparison uses one calculation thread
        "--config",
        "GDAL_NUM_THREADS",
        "1",
    ]
    if gdal_cachemax_mb is not None:
        command.extend(("--config", "GDAL_CACHEMAX", str(gdal_cachemax_mb)))
    command.extend(
        [
            "-zfield",
            zfield,
            "-a",
            algorithm_definition,
            "-txe",
            str(left),
            str(right),
            "-tye",
            str(bottom),
            str(top),
            "-outsize",
            str(width),
            str(height),
            "-ot",
            "Float64",
        ]
    )
    if output_crs is not None:
        command.extend(("-a_srs", output_crs))
    if creation_options is not None:
        command.extend(creation_options)
    command.extend(
        [
            "-l",
            layer,
            point_file,
            output_file,
        ]
    )
    return GdalCommand(command=command, output_file=output_file)


def build_gdal_command(
    operation: ComparisonOperation,
    config: BenchmarkConfig,
    raster_file: str,
    vector_file: str,
    point_file: str,
) -> GdalCommand:
    """Build one GDAL command with the same input and output grid as GeoUtils."""

    if config.directory is None:
        raise ValueError("GDAL comparison commands require an explicit output directory")

    # Every raster command writes the same tiled dimensions and bounded GDAL cache
    height, width = config.shape
    common_config = ["--config", "GDAL_CACHEMAX", str(config.gdal_cachemax_mb)]
    common_creation = ["-co", "TILED=YES", "-co", "BLOCKXSIZE=512", "-co", "BLOCKYSIZE=512"]

    if operation == "reproject":
        # Match the GeoUtils WGS84 to UTM zone 32N nearest-neighbor workflow
        output_file = os.path.join(config.directory, "output-gdal-reproject.tif")
        command = [
            _require_command("gdalwarp"),
            *common_config,
            "-overwrite",
            "-t_srs",
            "EPSG:32632",
            "-ts",
            str(width),
            str(height),
            "-r",
            "near",
            "-ot",
            "Float32",
            "-dstnodata",
            "-99999",
            "-et",
            "0",
            "-wm",
            "64",
            "-wo",
            "NUM_THREADS=1",
            "-wo",
            "XSCALE=1",
            "-wo",
            "YSCALE=1",
            *common_creation,
            raster_file,
            output_file,
        ]
        return GdalCommand(command, output_file)

    if operation == "polygonize":
        # The patterned raster lets feature count validate the complete vector output
        output_file = os.path.join(config.directory, "output-gdal-polygonize.gpkg")
        command = [
            _require_command("gdal_polygonize.py"),
            *common_config,
            raster_file,
            output_file,
            "polygons",
            "value",
            "-f",
            "GPKG",
            "-b",
            "1",
        ]
        return GdalCommand(command, output_file)

    if operation == "rasterize":
        # Byte burn values match the GeoUtils binary rasterization output
        output_file = os.path.join(config.directory, "output-gdal-rasterize.tif")
        command = [
            _require_command("gdal_rasterize"),
            *common_config,
            "-burn",
            "1",
            "-init",
            "0",
            "-ot",
            "Byte",
            "-te",
            "7",
            "45",
            "8",
            "46",
            "-ts",
            str(width),
            str(height),
            "-a_srs",
            "EPSG:4326",
            "-l",
            "source-vector",
            *common_creation,
            vector_file,
            output_file,
        ]
        return GdalCommand(command, output_file)

    if operation == "grid":
        # Select the equivalent GDAL algorithm for each representative GeoUtils method
        output_file = os.path.join(config.directory, "output-gdal-grid.tif")
        algorithms: dict[str, GdalGridAlgorithm] = {
            "nearest": "nearest",
            "linear": "linear",
            "idw": "invdist",
            "mean": "average",
            "average": "average",
            "minimum": "minimum",
            "min": "minimum",
            "maximum": "maximum",
            "max": "maximum",
            "range": "range",
            "count": "count",
            "average_distance": "average_distance",
            "average_distance_pts": "average_distance_pts",
        }
        if config.grid_resampling not in algorithms:
            raise ValueError(f"GDAL has no matching gridding method for {config.grid_resampling!r}")

        # A zero radius means unlimited nearest support and disables linear fallback outside the triangulation
        if config.grid_resampling in ("nearest", "linear"):
            radius = (0.0, 0.0)
        else:
            pixel_width = 1 / width
            pixel_height = 1 / height
            radius = (
                config.grid_dist_nodata_pixel * pixel_width,
                config.grid_dist_nodata_pixel * pixel_height,
            )
        return build_gdal_grid_command(
            point_file,
            output_file,
            algorithm=algorithms[config.grid_resampling],
            bounds=(7.0, 45.0, 8.0, 46.0),
            shape=config.shape,
            radius=radius,
            min_points=1,
            distance_power=2,
            gdal_cachemax_mb=config.gdal_cachemax_mb,
            output_crs="EPSG:4326",
            creation_options=common_creation,
        )

    raise ValueError(f"Unsupported GDAL comparison operation: {operation}")
