"""Minimal tests for benchmark workflows, reports and GDAL commands."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from benchmarks.asv_suite import comparisons as benchmark_comparisons
from benchmarks.asv_suite.comparisons import (
    BENCHMARK_CASE_BY_CLASS,
    BENCHMARK_CASES,
    COMPARISONS,
    EXTERNAL_REFERENCE_CASE_BY_CLASS,
)
from benchmarks.asv_suite.render_results import (
    COMPARISON_REPORT_DIRECTORY,
    DOCUMENTATION_DATA,
    DOCUMENTATION_MEMORY_PLOT,
    DOCUMENTATION_TIME_PLOT,
    PERFORMANCE_CHANGE_REPORT,
    SCALING_REPORT_PAGE,
    _PreviewResult,
    _render_preview,
    _select_complete_result,
    render_documentation_snapshot,
)
from benchmarks.gdal_comparison.commands import (
    COMPARISON_OPERATIONS,
    build_gdal_command,
)
from benchmarks.workflows.grouped_reference import (
    compute_grouped_reference,
    prepare_grouped_reference,
)
from benchmarks.workflows.runner import BenchmarkConfig
from benchmarks.workflows.variography import (
    prepare_pair_pointcloud,
    prepare_pair_raster,
    prepare_variogram_pairs,
)
from geoutils import Variogram


class TestComparisonReport:
    """
    Test module for benchmark registration and report generation.

    Those are minimal tests to ensure changes to benchmarks/ don't break the routines, even if ASV can do quick checks,
    it's easier to have a detailed traceback here through Pytest.
    We especially tests our custom routines/rendering for the benchmark webpage, comparisons to external refs (e.g.
    GDAL CLI), and across variables (raster/point size, chunk size, etc) and categories (e.g. eager/Dask/MP, method,
    etc) of interest.
    """

    def test_benchmark_registry__generated_classes_exist(self) -> None:
        """Checks that every registered benchmark and plotted series has a generated ASV class."""

        # Gather the GeoUtils/external classes registered for ASV and the classes referenced by report plots
        registered = set(BENCHMARK_CASE_BY_CLASS) | set(EXTERNAL_REFERENCE_CASE_BY_CLASS)
        plotted = {class_name for comparison in COMPARISONS for _, class_name in comparison.series}

        # Importing comparisons.py should create every class needed by ASV and the report
        assert BENCHMARK_CASES and COMPARISONS
        assert plotted <= registered
        assert all(hasattr(benchmark_comparisons, class_name) for class_name in registered)

    def test_render_preview__essential_files(self, tmp_path: Path) -> None:
        """Checks that preview rendering writes the main pages, data exports and plots."""

        pytest.importorskip("matplotlib")

        # Render the complete preview from the small fake ASV result bundled with the renderer
        _render_preview(tmp_path)
        report_directory = tmp_path / COMPARISON_REPORT_DIRECTORY

        # Keep this list to files used directly by the published benchmark site or downloaded result data
        expected_files = (
            tmp_path / "index.html",
            report_directory / "index.html",
            report_directory / SCALING_REPORT_PAGE,
            report_directory / "comparisons.csv",
            report_directory / "comparisons.json",
            report_directory / PERFORMANCE_CHANGE_REPORT,
        )
        assert all(path.is_file() for path in expected_files)
        assert any(report_directory.glob("*.svg"))

        # The JSON export must contain measurements rather than only an empty report shell
        payload = json.loads((report_directory / "comparisons.json").read_text(encoding="utf-8"))
        assert payload["measurements"]

    def test_select_complete_result__skips_incomplete_latest_run(self) -> None:
        """Checks that report selection falls back when the latest ASV result is incomplete."""

        # Make the second result newer, then remove one measurement that the report needs
        complete = _PreviewResult()
        complete.started_at = {"benchmark": 1}
        incomplete = _PreviewResult()
        incomplete.started_at = {"benchmark": 2}
        incomplete.values.pop(next(iter(incomplete.values)))

        # The earlier complete result should still be usable for report generation
        assert _select_complete_result((complete, incomplete)) is complete

    def test_render_documentation_snapshot__essential_files(self, tmp_path: Path) -> None:
        """Checks that documentation rendering writes its two plots and numeric data."""

        pytest.importorskip("matplotlib")

        # Render the documentation files from the same small result used by preview mode
        records = render_documentation_snapshot(_PreviewResult(), tmp_path)

        # Both graphics and their JSON source are needed when benchmark results are updated in the docs
        assert records
        assert (tmp_path / DOCUMENTATION_TIME_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_DATA).is_file()


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestGroupedReferenceChunked:
    """Test module for running grouped benchmark workflows with each execution path."""

    @pytest.mark.parametrize("implementation", ["geoutils", "flox"])
    @pytest.mark.parametrize("execution_mode", ["eager", "dask"])
    def test_grouped_reference__runs(
        self,
        implementation: Literal["geoutils", "flox"],
        execution_mode: Literal["eager", "dask"],
    ) -> None:
        """Checks that GeoUtils and Flox return a complete table from eager and Dask inputs."""

        # Flox is an optional benchmark reference, so leave its two cases out of the base test environment
        if implementation == "flox":
            pytest.importorskip("flox")

        # Prepare four groups from small arrays, then run the same entry point used by the benchmark classes
        inputs = prepare_grouped_reference(16, 2, "interleaved", execution_mode)
        result = compute_grouped_reference(*inputs, implementation=implementation)

        # Four groups x two value arrays each contain count, mean and standard deviation
        assert result.shape == (4, 6)
        assert np.array_equal(result.index, np.arange(4))
        assert np.isfinite(result.to_numpy()).all()

    def test_grouped_reference__multiprocessing_benchmark_runs(self) -> None:
        """Checks that one generated multiprocessing benchmark can set up, run and clean up."""

        # Find the generated GeoUtils case used beside the Flox raster-size comparison
        case = next(
            case
            for case in BENCHMARK_CASES
            if case.comparison_group == "grouped-flox-raster-size" and case.execution_mode == "multiprocessing"
        )
        benchmark = getattr(benchmark_comparisons, case.benchmark_class)()

        # Use a small raster but follow ASV's normal setup/run/teardown order, including its real worker process
        parameter = 32
        try:
            benchmark.setup(parameter)
            benchmark.time_operation(parameter)
        finally:
            benchmark.teardown(parameter)


class TestVariographyWorkflows:
    """Test module for running the pair and variogram benchmark workflows."""

    def test_variogram_pairs__runs(self) -> None:
        """Checks that prepared pairs can be reduced into populated variogram bins."""

        # Prepare complete pairs across the same distance range used by the benchmark
        pairs = prepare_variogram_pairs(200)
        edges = np.geomspace(1, 1024, 9)

        # Build a variogram through the public API and check that every prepared pair reaches one bin
        result = Variogram.from_pairs(
            pairs,
            bins=edges,
            estimator=lambda differences: float(np.mean(differences**2) / 2),
        )
        assert pairs.sizes["pair"] == 200
        assert result.counts.sum() == 200
        assert np.isfinite(result.semivariance).all()

    def test_pair_pointcloud__runs(self) -> None:
        """Checks that the point benchmark fixture produces the requested number of usable pairs."""

        # Build a small irregular point cloud and draw pairs with the benchmark's fixed random seed
        points = prepare_pair_pointcloud(200)
        pairs = points.pairsample(n_pairs=20, min_distance=1, max_distance=10, random_state=42)

        # Pair values have two endpoints and a finite distance for each requested pair
        assert pairs.sizes == {"pair": 20, "endpoint": 2}
        assert np.isfinite(pairs["value"]).all()
        assert np.isfinite(pairs["distance"]).all()


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestPairRasterChunked:
    """Test module for running raster pair sampling with Dask input."""

    def test_pair_raster__dask_matches_eager(self) -> None:
        """Checks that the Dask fixture stays lazy and returns the same pairs as the eager fixture."""

        import dask.array as da

        # Prepare the same raster in memory and as Dask data without loading the Dask values
        eager = prepare_pair_raster(32, "eager")
        lazy = prepare_pair_raster(32, "dask")
        assert isinstance(lazy.data, da.Array)
        assert not lazy._obj._in_memory

        # Draw the same 20 pairs from both inputs and compare the complete public pair datasets
        options = {"n_pairs": 20, "min_distance": 1, "max_distance": 16, "random_state": 42}
        expected = eager.pairsample(**options)
        result = lazy.pairsample(**options)
        assert not result.chunks
        xr.testing.assert_equal(result, expected)


class TestGdalCommands:
    """Test module for building GDAL commands used by external benchmark comparisons."""

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command__essential_arguments(
        self, operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that every GDAL comparison names its executable, input, output and cache size."""

        # Return command names directly because this smoke test builds arguments without running GDAL
        monkeypatch.setattr("benchmarks.gdal_comparison.commands._require_command", lambda name: name)
        config = BenchmarkConfig(shape=(64, 96), chunks=(32, 32), directory=str(tmp_path))

        # Build the command with one recognizable source path for each supported input type
        comparison = build_gdal_command(
            operation,  # type: ignore[arg-type]
            config,
            raster_file="source-raster.tif",
            vector_file="source-vector.gpkg",
            point_file="source-points.gpkg",
        )
        command = comparison.command

        # These fields are enough to catch a command wired to the wrong tool, source, output or cache setting
        expected_executable = {
            "reproject": "gdalwarp",
            "polygonize": "gdal_polygonize.py",
            "rasterize": "gdal_rasterize",
            "grid": "gdal_grid",
        }[operation]
        expected_source = {
            "reproject": "source-raster.tif",
            "polygonize": "source-raster.tif",
            "rasterize": "source-vector.gpkg",
            "grid": "source-points.gpkg",
        }[operation]
        cache_index = command.index("GDAL_CACHEMAX")
        assert command[0] == expected_executable
        assert expected_source in command
        assert comparison.output_file in command
        assert command[cache_index + 1] == str(config.gdal_cachemax_mb)
