"""Execute equivalent GDAL commands and collect their validated measurements."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import geopandas as gpd

from benchmarks.gdal_comparison.commands import (
    ComparisonOperation,
    GdalCommand,
    build_gdal_command,
)
from benchmarks.workflows.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    ProfiledResult,
    read_raster_center,
)
from geoutils.profiler import ProfileMetrics, profile_call


@dataclass(frozen=True)
class GdalResult(ProfiledResult):
    """Store one completed GDAL result together with its process memory."""

    value: float
    metrics: ProfileMetrics
    output_file: str


def read_comparison_value(operation: ComparisonOperation, output_file: str) -> float:
    """Read a small correctness fingerprint from one complete GDAL output."""

    if operation == "polygonize":
        # Feature count validates the complete vector output without retaining it
        return float(len(gpd.read_file(output_file)))

    # Raster workflows use a central pixel that is one for every deterministic fixture
    return read_raster_center(output_file)


class GdalRunner:
    """Run one GDAL operation against sources prepared by the shared workflow."""

    def __init__(
        self,
        operation: ComparisonOperation,
        config: BenchmarkConfig,
        sources: BenchmarkRunner,
    ) -> None:
        """Prepare a command without executing or measuring the operation."""

        self.operation = operation
        self.config = config

        # Polygonization uses the patterned source while other raster operations stay constant
        raster_file = sources.polygon_raster_file if operation == "polygonize" else sources.raster_file
        self.comparison: GdalCommand = build_gdal_command(
            operation,
            config,
            raster_file=raster_file,
            vector_file=sources.vector_file,
            point_file=sources.point_file,
        )

    def close(self) -> None:
        """Provide the same cleanup interface as the GeoUtils runner."""

        # The ASV comparison owns the shared directory and removes it after all runners close

    def _execute(self) -> float:
        """Execute the command and return a small fingerprint of its complete output."""

        # Repeated measurements reuse a stable path but never an earlier output
        if os.path.isfile(self.comparison.output_file):
            os.remove(self.comparison.output_file)
        completed = subprocess.run(self.comparison.command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"GDAL command failed with status {completed.returncode}: "
                f"{self.comparison.command}\n{completed.stderr}"
            )

        # Reading a fingerprint ensures command completion includes a usable result
        value = read_comparison_value(self.operation, self.comparison.output_file)
        expected = self.config.polygon_regions_per_axis**2 if self.operation == "polygonize" else 1
        if value != expected:
            raise RuntimeError(f"Unexpected GDAL {self.operation} validation value: {value}")
        return value

    def run(self) -> GdalResult:
        """Execute one command while sampling the benchmark process and GDAL child."""

        # The same process-tree boundary is used by every ASV implementation
        value, metrics = profile_call(
            self._execute,
            interval=self.config.profile_interval,
            dask=False,
            include_children=True,
        )
        return GdalResult(float(value), metrics, self.comparison.output_file)
