"""Test custom benchmark reports and the definitions of GDAL comparison commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.asv_suite.comparisons import COMPARISONS
from benchmarks.asv_suite.render_results import (
    COMPARISON_REPORT_DIRECTORY,
    DOCUMENTATION_DATA,
    DOCUMENTATION_MEMORY_PLOT,
    DOCUMENTATION_TIME_PLOT,
    _select_complete_result,
    collect_implementation_measurements,
    render_comparisons,
    render_documentation_snapshot,
)
from benchmarks.gdal_comparison.commands import (
    COMPARISON_OPERATIONS,
    build_gdal_command,
)
from benchmarks.workflows.runner import BenchmarkConfig


class TestComparisonReport:
    """Check that saved ASV comparisons become reusable numeric reports and plots."""

    class FakeResult:
        """Provide the small part of the ASV Results interface used by the renderer."""

        commit_hash = "0123456789abcdef"
        date = 1_700_000_000_000
        params = {"machine": "test-machine", "cpu": "test-cpu"}
        started_at = {"benchmark": 1_700_000_001_000}
        _env_name = "existing-py3.12"

        def __init__(self) -> None:
            """Build matching time and memory values for every comparison class."""

            self.values = {}
            self.parameters = {}
            for comparison in COMPARISONS:
                for implementation_index, (_, class_name) in enumerate(comparison.series, start=1):
                    # Test parameters are stored as repr strings like real ASV results
                    parameters = [["256", "512", "1024"]]
                    prefix = f"asv_suite.comparisons.{class_name}"
                    self.parameters[f"{prefix}.time_operation"] = parameters
                    self.parameters[f"{prefix}.track_end_to_end_time_s"] = parameters
                    self.parameters[f"{prefix}.track_peak_process_tree_rss_mb"] = parameters
                    self.values[f"{prefix}.time_operation"] = [
                        implementation_index * value / 1000 for value in (1, 2, 4)
                    ]
                    self.values[f"{prefix}.track_end_to_end_time_s"] = [
                        implementation_index * value / 100 for value in (1, 2, 4)
                    ]
                    self.values[f"{prefix}.track_peak_process_tree_rss_mb"] = [
                        implementation_index * value for value in (100, 110, 120)
                    ]

        def get_all_result_keys(self) -> Any:
            """Return available benchmark keys like ASV Results."""

            return self.values.keys()

        def get_result_params(self, key: str) -> list[list[str]]:
            """Return the one stored parameter dimension."""

            return self.parameters[key]

        def get_result_value(self, key: str, params: list[list[str]]) -> list[float]:
            """Return measurements already aligned to the requested parameters."""

            assert params == self.parameters[key]
            return self.values[key]

    def test_collect_implementation_measurements(self) -> None:
        """Pair both time boundaries and peak RAM for every implementation."""

        # Every declared line and parameter must remain an independent numeric record
        records = collect_implementation_measurements(self.FakeResult())
        expected_records = sum(len(comparison.series) * 3 for comparison in COMPARISONS)
        assert len(records) == expected_records
        assert {record.comparison for record in records} == {comparison.slug for comparison in COMPARISONS}
        assert {record.implementation for record in records} == {
            "Eager",
            "Dask",
            "Multiprocessing",
            "Eager (Numba)",
            "SciPy",
            "Numba",
            "GDAL",
        }
        assert all(isinstance(record.parameter, int) for record in records)

    def test_select_complete_result__skip_incomplete(self) -> None:
        """Keep the latest complete result when a newer ASV run was interrupted."""

        complete = self.FakeResult()
        complete.started_at = {"benchmark": 1_700_000_001_000}
        incomplete = self.FakeResult()
        incomplete.started_at = {"benchmark": 1_700_000_002_000}

        # Removing one series reproduces an interrupted comparison run
        incomplete.values.pop(next(iter(incomplete.values)))

        assert _select_complete_result((complete, incomplete)) is complete

    def test_render_comparisons(self, tmp_path: Path) -> None:
        """Write one site index, numeric files and one SVG per comparison."""

        pytest.importorskip("matplotlib")

        # The root remains the single entry point for local artifacts and GitHub Pages
        render_comparisons(self.FakeResult(), tmp_path)
        assert (tmp_path / "index.html").is_file()
        assert "comparisons/index.html" in (tmp_path / "index.html").read_text(encoding="utf-8")
        report_directory = tmp_path / COMPARISON_REPORT_DIRECTORY
        assert (report_directory / "index.html").is_file()
        assert (report_directory / "comparisons.json").is_file()
        assert (report_directory / "comparisons.csv").is_file()
        assert {path.name for path in report_directory.glob("*.svg")} == {
            f"{comparison.slug}.svg" for comparison in COMPARISONS
        }

    def test_render_documentation_snapshot(self, tmp_path: Path) -> None:
        """Write two concise graphics with their exact measurements and provenance."""

        pytest.importorskip("matplotlib")

        # Documentation files derive from the same records as the detailed ASV website
        render_documentation_snapshot(self.FakeResult(), tmp_path)
        assert (tmp_path / DOCUMENTATION_TIME_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_DATA).is_file()

        snapshot = json.loads((tmp_path / DOCUMENTATION_DATA).read_text(encoding="utf-8"))
        assert snapshot["metadata"]["commit"] == "0123456789abcdef"
        assert snapshot["metadata"]["machine"]["machine"] == "test-machine"


class TestGdalCommands:
    """Check the settings that keep GDAL and GeoUtils benchmark operations comparable."""

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command(self, operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Build every comparison with fixed inputs, resources and output dimensions."""

        # Command construction does not need the executable or a profiled benchmark run
        monkeypatch.setattr("benchmarks.gdal_comparison.commands._require_command", lambda name: name)
        config = BenchmarkConfig(shape=(64, 96), chunks=(32, 32), directory=str(tmp_path))
        comparison = build_gdal_command(
            operation,  # type: ignore[arg-type]
            config,
            raster_file="source-raster.tif",
            vector_file="source-vector.gpkg",
            point_file="source-points.gpkg",
        )
        command = comparison.command

        # Every operation uses the same bounded GDAL cache and writes its declared output
        cache_index = command.index("GDAL_CACHEMAX")
        assert command[cache_index + 1] == str(config.gdal_cachemax_mb)
        assert comparison.output_file in command

        if operation == "reproject":
            # Match the GeoUtils target grid, exact transform and single calculation thread
            size_index = command.index("-ts")
            assert command[0] == "gdalwarp"
            assert command[size_index + 1 : size_index + 3] == ["96", "64"]
            assert command[command.index("-et") + 1] == "0"
            assert "NUM_THREADS=1" in command
            assert "XSCALE=1" in command
            assert "YSCALE=1" in command
        elif operation == "polygonize":
            assert command[0] == "gdal_polygonize.py"
            assert command[command.index("-b") + 1] == "1"
            assert "source-raster.tif" in command
        elif operation == "rasterize":
            # Burn the same binary values over the same WGS84 extent and raster size
            size_index = command.index("-ts")
            assert command[0] == "gdal_rasterize"
            assert command[size_index + 1 : size_index + 3] == ["96", "64"]
            assert command[command.index("-burn") + 1] == "1"
            assert command[command.index("-init") + 1] == "0"
        else:
            # Gridding uses the equivalent nearest method, output grid and one calculation thread
            size_index = command.index("-outsize")
            algorithm = command[command.index("-a") + 1]
            thread_index = command.index("GDAL_NUM_THREADS")
            assert command[0] == "gdal_grid"
            assert command[size_index + 1 : size_index + 3] == ["96", "64"]
            assert algorithm.startswith("nearest:")
            assert command[thread_index + 1] == "1"
