"""Verify benchmark report generation and equivalent GDAL CLI command arguments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.asv_suite.comparisons import (
    BENCHMARK_CASE_BY_CLASS,
    BENCHMARK_CASES,
    CALCULATION_ENGINE_LABELS,
    COMPARISONS,
    EXECUTION_MODE_LABELS,
    EXTERNAL_REFERENCE_CASE_BY_CLASS,
    GDAL_CLI_LABEL,
    METHOD_LABELS,
    STRATEGY_LABELS,
)
from benchmarks.asv_suite.operations import FIXED_OPERATION_BENCHMARK_CASES
from benchmarks.asv_suite.render_results import (
    COMPARISON_REPORT_DIRECTORY,
    DOCUMENTATION_DATA,
    DOCUMENTATION_MEMORY_PLOT,
    DOCUMENTATION_TIME_PLOT,
    HISTORY_REPORT_PAGE,
    PERFORMANCE_CHANGE_REPORT,
    SCALING_REPORT_PAGE,
    _PreviewResult,
    _render_preview,
    _select_complete_result,
    collect_comparison_measurements,
    performance_change_markdown,
    render_comparisons,
    render_documentation_snapshot,
)
from benchmarks.gdal_comparison.commands import (
    COMPARISON_OPERATIONS,
    _warp_memory_limit_mb,
    build_gdal_command,
)
from benchmarks.workflows.registry import (
    OPERATION_METHODS,
    OPERATION_STRATEGIES,
    split_operation_case,
)
from benchmarks.workflows.runner import BenchmarkConfig


class TestComparisonReport:
    """Verify report behavior that ASV's quick measurement command does not cover."""

    def test_collect_comparison_measurements(self) -> None:
        """Checks that each saved plot series and parameter becomes one typed measurement row."""

        # Each result keeps independent dimensions instead of encoding them in one implementation label
        records = collect_comparison_measurements(_PreviewResult())
        expected_records = sum(len(comparison.series) * 3 for comparison in COMPARISONS)
        assert len(records) == expected_records
        assert {record.comparison for record in records} == {comparison.slug for comparison in COMPARISONS}
        comparison_by_slug = {comparison.slug: comparison for comparison in COMPARISONS}
        for record in records:
            comparison = comparison_by_slug[record.comparison]
            assert record.operation == comparison.operation
            if record.series_dimension != "method":
                assert record.method == comparison.method
            if record.external_reference is not None:
                assert record.series_label == GDAL_CLI_LABEL
                assert record.execution_mode is None
                assert record.calculation_engine is None
            else:
                assert record.execution_mode is not None
                if record.series_dimension == "execution_mode":
                    expected_label = EXECUTION_MODE_LABELS[record.execution_mode]
                elif record.series_dimension == "calculation_engine":
                    assert record.calculation_engine is not None
                    expected_label = CALCULATION_ENGINE_LABELS[record.calculation_engine]
                elif record.series_dimension == "strategy":
                    assert record.strategy is not None
                    expected_label = STRATEGY_LABELS[record.strategy]
                else:
                    assert record.method is not None
                    expected_label = METHOD_LABELS.get(record.method, record.method.replace("_", " ").title())
                assert record.series_label == expected_label
        assert all(isinstance(record.parameter, int) for record in records)

    def test_benchmark_dimension_registry(self) -> None:
        """Checks that generated cases follow the capability registry and one-axis coverage rules."""

        assert {comparison.parameter_label for comparison in COMPARISONS} == {
            "Size of raster (pixels per side)",
            "Size of chunks (pixels per side)",
            "Number of interpolated points",
            "Number of source points per axis",
            "Number of sampled values",
        }
        method_by_key = {
            (specification.operation, specification.method): specification for specification in OPERATION_METHODS
        }
        assert len(BENCHMARK_CASE_BY_CLASS) == len(BENCHMARK_CASES)
        assert not (set(BENCHMARK_CASE_BY_CLASS) & set(EXTERNAL_REFERENCE_CASE_BY_CLASS))
        series_classes = {class_name for comparison in COMPARISONS for _, class_name in comparison.series}
        worker_classes = {
            case.benchmark_class for case in BENCHMARK_CASES if case.comparison_group == "worker-integration"
        }
        assert series_classes | worker_classes == set(BENCHMARK_CASE_BY_CLASS) | set(EXTERNAL_REFERENCE_CASE_BY_CLASS)

        strategy_by_operation = {
            operation: {
                specification.strategy for specification in OPERATION_STRATEGIES if specification.operation == operation
            }
            for operation in {specification.operation for specification in OPERATION_STRATEGIES}
        }
        for case in BENCHMARK_CASES:
            if case.calculation_engine is not None:
                specification = method_by_key[(case.operation, case.method)]
                assert case.calculation_engine in specification.calculation_engines
            if case.strategy is not None:
                assert case.execution_mode in ("dask", "multiprocessing")
                assert case.strategy in strategy_by_operation[case.operation]

        # Every plot changes only its declared series dimension
        for comparison in COMPARISONS:
            cases = [
                BENCHMARK_CASE_BY_CLASS[class_name]
                for label, class_name in comparison.series
                if label != GDAL_CLI_LABEL
            ]
            assert {case.operation for case in cases} == {comparison.operation}
            if comparison.series_dimension != "method":
                assert {case.method for case in cases} == {comparison.method}
            if comparison.series_dimension == "execution_mode":
                assert {case.calculation_engine for case in cases} == {comparison.calculation_engine}
                assert {case.strategy for case in cases if case.execution_mode != "eager"} == {comparison.strategy}
            elif comparison.series_dimension == "calculation_engine":
                assert {case.execution_mode for case in cases} == {comparison.execution_mode}
                specification = method_by_key[(comparison.operation, comparison.method)]
                assert {case.calculation_engine for case in cases} == set(specification.calculation_engines)
            elif comparison.series_dimension == "strategy":
                assert {case.execution_mode for case in cases} == {comparison.execution_mode}
                assert {case.calculation_engine for case in cases} == {comparison.calculation_engine}
                assert {case.strategy for case in cases} == strategy_by_operation[comparison.operation]
            else:
                assert {case.execution_mode for case in cases} == {comparison.execution_mode}
                assert {case.calculation_engine for case in cases} == {comparison.calculation_engine}
                assert {case.strategy for case in cases} == {comparison.strategy}

        # Numba scaling stays eager, while fixed sentinels cover every kernel in both worker modes
        gridding_cases = [case for case in BENCHMARK_CASES if case.comparison_group == "gridding-raster-size"]
        assert all(case.calculation_engine == "scipy" for case in gridding_cases if case.execution_mode != "eager")
        worker_cases = [case for case in BENCHMARK_CASES if case.comparison_group == "worker-integration"]
        assert {(case.method, case.execution_mode) for case in worker_cases} == {
            (method, execution_mode)
            for method in ("nearest", "idw", "mean")
            for execution_mode in ("dask", "multiprocessing")
        }
        assert {case.calculation_engine for case in worker_cases} == {"numba"}
        assert {(case.method, case.execution_mode) for case in worker_cases if case.pr_check} == {
            ("idw", "dask"),
            ("mean", "multiprocessing"),
        }

        # The fixed sweep retains only operations that have no scaling comparison
        assert len(FIXED_OPERATION_BENCHMARK_CASES) == 9
        assert {split_operation_case(case)[1] for case in FIXED_OPERATION_BENCHMARK_CASES} == {
            "crop",
            "translate",
            "copy",
            "statistics",
            "subsample",
            "write",
            "create_mask",
        }

    def test_collect_comparison_measurements__omit_skipped(self) -> None:
        """Checks that an ASV skip marker produces no report metric for that parameter combination."""

        result = _PreviewResult()
        comparison = COMPARISONS[0]
        series_label, class_name = comparison.series[0]
        prefix = f"asv_suite.comparisons.{class_name}"
        for method in ("time_operation", "track_end_to_end_time_s", "track_peak_process_tree_mem_mb"):
            result.values[f"{prefix}.{method}"][0] = float("nan")

        records = collect_comparison_measurements(result)
        assert not any(
            record.comparison == comparison.slug and record.series_label == series_label and record.parameter == 256
            for record in records
        )

    def test_select_complete_result__skip_incomplete(self) -> None:
        """Checks that result selection skips a newer incomplete run and returns the latest complete run."""

        complete = _PreviewResult()
        complete.started_at = {"benchmark": 1_700_000_001_000}
        incomplete = _PreviewResult()
        incomplete.started_at = {"benchmark": 1_700_000_002_000}

        # Removing one metric key makes the newer result incomplete
        incomplete.values.pop(next(iter(incomplete.values)))

        assert _select_complete_result((complete, incomplete)) is complete

    def test_render_comparisons(self, tmp_path: Path) -> None:
        """Checks that rendering a complete result writes navigation, numeric exports and every plot."""

        pytest.importorskip("matplotlib")

        # The root summary must lead to both custom views and the native history
        asv_directory = tmp_path / "asv"
        asv_directory.mkdir()
        (asv_directory / "index.html").write_text("Native ASV report", encoding="utf-8")
        render_comparisons(_PreviewResult(), tmp_path)
        assert (tmp_path / "index.html").is_file()
        root_page = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "comparisons/index.html" in root_page
        assert f"comparisons/{SCALING_REPORT_PAGE}" in root_page
        assert HISTORY_REPORT_PAGE in root_page
        assert "GeoUtils performance benchmarks" in root_page
        assert 'class="data-note"' in root_page
        assert "16.8 MB as Float32" in root_page
        assert "before object overhead" in root_page
        assert "GeoUtils calculation engine" in root_page
        assert "GeoUtils execution mode" in root_page
        assert "External reference" in root_page
        for option in ("Rasterio/GDAL", "SciPy", "Numba", "GDAL CLI", "Eager", "Dask", "Multiprocessing"):
            assert f">{option}</th>" in root_page
        assert '<span class="metric-label">Fastest</span>' not in root_page
        assert "MB" in root_page
        assert "Filtering — Mean" in root_page
        assert "Filtering — Circular mean" not in root_page
        assert '<span class="workload-list"><span>2,601 source polygon features</span>' in root_page
        assert "<span>2,048 × 2,048 output raster</span>" in root_page
        assert '<span class="workload-list"><span>2,048 × 2,048 output raster</span>' in root_page
        assert "<span>17 × 17 source points</span>" in root_page
        assert '<span class="workload-list"><span>2,048 × 2,048 raster</span>' in root_page
        assert "<span>1,024 × 1,024 chunks</span>" in root_page
        assert "4,096 × 4,096" not in root_page
        assert "441 disconnected raster regions" in root_page
        assert "producing 441 polygon features" not in root_page
        assert "× GDAL CLI" not in root_page
        assert 'class="external-cell"' in root_page
        assert 'class="workload-heading"' in root_page
        assert "All options run without parallelism: one computational process using one thread" in root_page
        assert "Prepared inputs through completed output" in root_page
        assert "data larger than memory" in root_page
        assert "Ratios" not in root_page
        assert '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils calculation engine</th>' in root_page
        assert '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils execution mode</th>' in root_page
        operation_groups = (
            "Raster ⟶ Raster",
            "Raster ⟶ Point",
            "Raster ⟶ Vector",
            "Point ⟶ Raster",
            "Vector ⟶ Raster",
        )
        assert all(f'scope="rowgroup">{group}</th>' in root_page for group in operation_groups)
        assert [root_page.index(group) for group in operation_groups] == sorted(
            root_page.index(group) for group in operation_groups
        )
        history_page = (tmp_path / HISTORY_REPORT_PAGE).read_text(encoding="utf-8")
        assert 'aria-current="page">Performance history</a>' in history_page
        assert 'href="index.html">Summary</a>' in history_page
        assert 'href="comparisons/index.html">Options</a>' in history_page
        assert f'href="comparisons/{SCALING_REPORT_PAGE}">Scaling</a>' in history_page
        assert '<iframe class="history-frame" src="asv/index.html"' in history_page
        assert "Native ASV report" not in history_page
        report_directory = tmp_path / COMPARISON_REPORT_DIRECTORY
        assert (report_directory / "index.html").is_file()
        assert (report_directory / SCALING_REPORT_PAGE).is_file()
        assert (report_directory / "comparisons.json").is_file()
        assert (report_directory / "comparisons.csv").is_file()
        assert {path.name for path in report_directory.glob("*.svg")} == {
            f"{comparison.slug}.svg" for comparison in COMPARISONS
        }

        # The option page defines each choice plainly and keeps the external command separate
        comparison_page = (report_directory / "index.html").read_text(encoding="utf-8")
        for heading in (
            "Execution modes",
            "Calculation engines",
            "Operation methods",
            "Chunk strategies",
        ):
            assert heading in comparison_page
        assert "Compare GeoUtils options" in comparison_page
        assert "changes one GeoUtils option at a common workload" in comparison_page
        assert "GDAL CLI is shown separately" in comparison_page
        assert '<span class="metric-label">Time</span>' in comparison_page
        assert '<span class="metric-label">Memory</span>' in comparison_page
        assert "Ratios" in comparison_page
        assert comparison_page.count('class="operation-group"') >= len(operation_groups)
        assert 'class="group-heading" colspan="3" scope="colgroup">GeoUtils calculation engine' in comparison_page
        assert 'class="group-heading" colspan="3" scope="colgroup">GeoUtils execution mode' in comparison_page
        assert 'class="choice-table"' in comparison_page
        assert comparison_page.count('class="workload-heading"') == 2
        assert '<span class="workload-list"><span>2,048 × 2,048 source raster</span>' in comparison_page
        assert "<span>16,384 interpolated points</span>" in comparison_page
        assert "4,096 × 4,096" not in comparison_page

        # The scaling page groups collapsible plots by the input that grows
        scaling_page = (report_directory / SCALING_REPORT_PAGE).read_text(encoding="utf-8")
        for heading in (
            "Scaling with raster size",
            "Scaling with chunk size",
            "Scaling with the number of interpolated points",
            "Scaling with the number of source points",
            "Scaling with the number of sampled values",
        ):
            assert heading in scaling_page
        assert 'class="plot-toc"' in scaling_page
        assert "Each section varies one input size while other settings stay fixed" in scaling_page
        assert "Select a comparison to show or hide its plot" in scaling_page
        assert "Operation only" in scaling_page
        assert "Ratios" not in scaling_page
        assert "EPSG:4326" in scaling_page
        assert "EPSG:32632" in scaling_page
        assert all(f'id="{comparison.slug}"' in scaling_page for comparison in COMPARISONS)
        assert all(f'href="#{comparison.slug}"' in scaling_page for comparison in COMPARISONS)

    def test_render_preview(self, tmp_path: Path) -> None:
        """Checks that preview mode writes a complete custom website without a native ASV history."""

        pytest.importorskip("matplotlib")

        _render_preview(tmp_path)
        root_page = (tmp_path / "index.html").read_text(encoding="utf-8")
        assert "comparisons/index.html" in root_page
        assert f"comparisons/{SCALING_REPORT_PAGE}" in root_page
        assert HISTORY_REPORT_PAGE not in root_page
        assert not (tmp_path / HISTORY_REPORT_PAGE).exists()
        assert (tmp_path / COMPARISON_REPORT_DIRECTORY / PERFORMANCE_CHANGE_REPORT).is_file()

    def test_performance_change_markdown(self, tmp_path: Path) -> None:
        """Checks that changes use each revision's GDAL CLI time and only comparisons present in both."""

        pytest.importorskip("matplotlib")
        baseline = _PreviewResult()
        baseline.commit_hash = "baseline"
        current = _PreviewResult()
        current.commit_hash = "current"

        # Doubling each unique baseline GeoUtils result should produce a twofold normalized improvement
        baseline_keys = set()
        for comparison in COMPARISONS:
            for series_label, class_name in comparison.series:
                if series_label == GDAL_CLI_LABEL:
                    continue
                key = f"asv_suite.comparisons.{class_name}.track_end_to_end_time_s"
                baseline_keys.add(key)
        for key in baseline_keys:
            baseline.values[key] = [2 * value for value in baseline.values[key]]

        # Removing baseline gridding simulates a comparison introduced by the current revision
        added_comparison = next(comparison for comparison in COMPARISONS if comparison.slug == "gridding-raster-size")
        for _, class_name in added_comparison.series:
            prefix = f"asv_suite.comparisons.{class_name}"
            for method in ("time_operation", "track_end_to_end_time_s", "track_peak_process_tree_mem_mb"):
                baseline.values.pop(f"{prefix}.{method}")
                baseline.parameters.pop(f"{prefix}.{method}")

        markdown = performance_change_markdown(baseline, current)
        assert "| Eager |" in markdown
        assert "| Dask |" in markdown
        assert "| Multiprocessing |" in markdown
        assert markdown.count("2.00× faster") >= 3
        assert "Nearest gridding" not in markdown

        render_comparisons(current, tmp_path, baseline_result=baseline)
        report = tmp_path / COMPARISON_REPORT_DIRECTORY / PERFORMANCE_CHANGE_REPORT
        assert report.read_text(encoding="utf-8") == markdown
        assert PERFORMANCE_CHANGE_REPORT in (tmp_path / COMPARISON_REPORT_DIRECTORY / "README.md").read_text(
            encoding="utf-8"
        )

    def test_render_documentation_snapshot(self, tmp_path: Path) -> None:
        """Checks that a documentation snapshot writes both plots with their commit and machine metadata."""

        pytest.importorskip("matplotlib")

        # Both graphics and their machine/commit metadata must be written together
        render_documentation_snapshot(_PreviewResult(), tmp_path)
        assert (tmp_path / DOCUMENTATION_TIME_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_DATA).is_file()

        snapshot = json.loads((tmp_path / DOCUMENTATION_DATA).read_text(encoding="utf-8"))
        assert snapshot["metadata"]["commit"] == "preview-current"
        assert snapshot["metadata"]["machine"]["machine"] == "preview-machine"


class TestGdalCommands:
    """Verify GDAL CLI commands match GeoUtils inputs, outputs and resource limits."""

    def test_warp_memory_matches_execution_chunk(self) -> None:
        """Checks that GDAL can hold one 512 × 512 Float32 source/output chunk and their nodata masks."""

        assert _warp_memory_limit_mb(BenchmarkConfig(shape=(2048, 2048), chunks=(512, 512))) == 3

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command(self, operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Checks that each GDAL CLI command matches the GeoUtils input grid, output and resource limits."""

        # Stub executable lookup because this test inspects arguments without running the GDAL CLI
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

        # Every command must use the configured cache and include its expected output path
        cache_index = command.index("GDAL_CACHEMAX")
        assert command[cache_index + 1] == str(config.gdal_cachemax_mb)
        assert comparison.output_file in command

        if operation == "reproject":
            # Reprojection must match GeoUtils dimensions, exact transform and one thread
            size_index = command.index("-ts")
            assert command[0] == "gdalwarp"
            assert command[size_index + 1 : size_index + 3] == ["96", "64"]
            assert command[command.index("-et") + 1] == "0"
            assert command[command.index("-wm") + 1] == "1"
            assert "NUM_THREADS=1" in command
            assert "XSCALE=1" in command
            assert "YSCALE=1" in command
        elif operation == "polygonize":
            assert command[0] == "gdal_polygonize.py"
            assert command[command.index("-b") + 1] == "1"
            assert "source-raster.tif" in command
        elif operation == "rasterize":
            # Rasterization must burn the same binary mask into the same output grid
            size_index = command.index("-ts")
            assert command[0] == "gdal_rasterize"
            assert command[size_index + 1 : size_index + 3] == ["96", "64"]
            assert command[command.index("-burn") + 1] == "1"
            assert command[command.index("-init") + 1] == "0"
        else:
            # Gridding must use nearest interpolation on the same grid with one thread
            size_index = command.index("-outsize")
            algorithm = command[command.index("-a") + 1]
            thread_index = command.index("GDAL_NUM_THREADS")
            assert command[0] == "gdal_grid"
            assert command[size_index + 1 : size_index + 3] == ["96", "64"]
            assert algorithm.startswith("nearest:")
            assert command[thread_index + 1] == "1"
