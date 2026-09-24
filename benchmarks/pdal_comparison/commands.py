"""Build PDAL pipelines equivalent to raster point operations."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Any, Literal

from benchmarks.workflows.config import BenchmarkConfig

# Only these raster point operations have an equivalent PDAL pipeline for the external comparison
PdalComparisonOperation = Literal["subsample", "to_pointcloud"]
PDAL_COMPARISON_OPERATIONS: tuple[PdalComparisonOperation, ...] = ("subsample", "to_pointcloud")


@dataclass(frozen=True)
class PdalCommand:
    """Store one PDAL command, its pipeline definition and the file it must create."""

    command: list[str]
    pipeline_file: str
    output_file: str


def _require_command(name: str) -> str:
    """Return an installed PDAL executable or raise a clear environment error."""

    # Resolve executables once so subprocess never depends on shell parsing
    executable = shutil.which(name)
    if executable is None:
        raise RuntimeError(f"Required PDAL command is not installed: {name}")
    return executable


def build_pdal_command(
    operation: PdalComparisonOperation,
    config: BenchmarkConfig,
    raster_file: str,
) -> PdalCommand:
    """Build one PDAL raster-to-point pipeline and write its JSON definition."""

    if config.directory is None:
        raise ValueError("PDAL comparison commands require an explicit output directory")

    # Read every raster cell as a point with pixel-center coordinates and its band value as an attribute
    stages: list[dict[str, Any]] = [
        {
            "type": "readers.gdal",
            "filename": raster_file,
            "gdalopts": [f"GDAL_CACHEMAX={config.gdal_cachemax_mb}"],
        }
    ]
    if config.point_output_driver in ("LAS", "LAZ"):
        # Store the first raster band in the native elevation dimension used by the GeoUtils output
        stages.append({"type": "filters.ferry", "dimensions": "band_1=>Z"})
    if operation == "subsample":
        # Randomize the complete point view before keeping the requested fixed-size sample
        stages.extend(
            (
                {"type": "filters.randomize", "seed": 42},
                {"type": "filters.head", "count": config.subsample_size},
            )
        )

    # Write the same file-backed point format used by GeoUtils multiprocessing
    suffix = config.point_output_driver.lower()
    output_file = os.path.join(config.directory, f"output-pdal-{operation}.{suffix}")
    if config.point_output_driver == "GPKG":
        stages.append(
            {
                "type": "writers.ogr",
                "filename": output_file,
                "ogrdriver": "GPKG",
                "attr_dims": "all",
            }
        )
    else:
        stages.append(
            {
                "type": "writers.las",
                "filename": output_file,
                "compression": config.point_output_driver == "LAZ",
                "extra_dims": "all",
            }
        )

    pipeline_file = os.path.join(config.directory, f"pipeline-pdal-{operation}.json")
    with open(pipeline_file, "w", encoding="utf-8") as pipeline_stream:
        json.dump({"pipeline": stages}, pipeline_stream, indent=2)

    command = [_require_command("pdal"), "pipeline", pipeline_file]
    return PdalCommand(command=command, pipeline_file=pipeline_file, output_file=output_file)
