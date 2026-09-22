"""Execute equivalent PDAL pipelines and collect their validated measurements."""

from __future__ import annotations

import os
import pathlib
import subprocess
import warnings
from dataclasses import dataclass

import pyogrio

from benchmarks.pdal_comparison.commands import (
    PdalCommand,
    PdalComparisonOperation,
    build_pdal_command,
)
from benchmarks.workflows.runner import BenchmarkConfig, BenchmarkRunner, ProfiledResult
from geoutils.profiler import ProfileMetrics, profile_call


# Return PDAL measurements through the same result interface used by GeoUtils benchmark runs
@dataclass(frozen=True)
class PdalResult(ProfiledResult):
    """Store one completed PDAL result together with its process memory."""

    value: float
    metrics: ProfileMetrics
    output_file: str


def read_comparison_value(
    operation: PdalComparisonOperation,
    config: BenchmarkConfig,
    output_file: str,
) -> float:
    """Validate one complete PDAL point output and read one band value."""

    if pathlib.Path(output_file).suffix.lower() in (".las", ".laz"):
        from benchmarks.workflows.runner import read_point_file_sample

        count, value = read_point_file_sample(output_file, "Z")
        expected_count = config.subsample_size if operation == "subsample" else config.shape[0] * config.shape[1]
        if count != expected_count:
            raise RuntimeError(f"Unexpected PDAL {operation} output count: {count}")
        return value

    # Feature metadata verifies the complete output without loading every point back into memory
    # PDAL declares Z/M geometry, but only its band attribute is read here
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Measured \(M\) geometry types are not supported.*")
        info = pyogrio.read_info(output_file, force_feature_count=True)
    expected_count = config.subsample_size if operation == "subsample" else config.shape[0] * config.shape[1]
    if info["features"] != expected_count:
        raise RuntimeError(f"Unexpected PDAL {operation} output count: {info['features']}")

    # The constant raster makes any band value a sufficient value check regardless of the randomized point order
    fields = list(info.get("fields", []))
    if not fields:
        raise RuntimeError(f"PDAL {operation} output contains no raster band attribute")
    field = fields[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"Measured \(M\) geometry types are not supported.*")
        sample = pyogrio.read_dataframe(output_file, columns=[field], read_geometry=False, max_features=1)
    return float(sample[field].iloc[0])


# Adapt one PDAL pipeline to the setup, execution, validation and cleanup steps expected by ASV
class PdalRunner:
    """Run one PDAL operation against the raster prepared by the shared workflow."""

    def __init__(
        self,
        operation: PdalComparisonOperation,
        config: BenchmarkConfig,
        sources: BenchmarkRunner,
    ) -> None:
        """Prepare a pipeline without executing or measuring the operation."""

        self.operation = operation
        self.config = config
        self.comparison: PdalCommand = build_pdal_command(operation, config, raster_file=sources.raster_file)

    def close(self) -> None:
        """Provide the same cleanup interface as the GeoUtils runner."""

        # The ASV comparison owns the shared directory and removes it after all runners close

    def _execute(self) -> float:
        """Execute the pipeline and return a small fingerprint of its complete output."""

        # Repeated measurements reuse a stable path but never an earlier output
        if os.path.isfile(self.comparison.output_file):
            os.remove(self.comparison.output_file)
        completed = subprocess.run(self.comparison.command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"PDAL command failed with status {completed.returncode}: {self.comparison.command}\n{completed.stderr}"
            )

        # Reading one value ensures command completion includes a usable result
        value = read_comparison_value(self.operation, self.config, self.comparison.output_file)
        if value != self.config.raster_value:
            raise RuntimeError(f"Unexpected PDAL {self.operation} validation value: {value}")
        return value

    def run(self) -> PdalResult:
        """Execute one pipeline while sampling the benchmark process and PDAL child."""

        # The same process-tree boundary is used by every ASV implementation
        value, metrics = profile_call(
            self._execute,
            interval=self.config.profile_interval,
            dask=False,
            include_children=True,
        )
        return PdalResult(float(value), metrics, self.comparison.output_file)
