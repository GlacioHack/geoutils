"""Render saved ASV measurements for the benchmark website and user documentation."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.asv_suite.comparisons import (
    BENCHMARK_CASE_BY_CLASS,
    COMPARISONS,
    EXTERNAL_REFERENCE_CASE_BY_CLASS,
    GDAL_CLI_LABEL,
    Comparison,
    ComparisonDimension,
    ExternalReference,
)
from benchmarks.workflows.registry import (
    CalculationEngine,
    ExecutionMode,
    OperationName,
    OperationStrategyName,
)

COMPARISON_BENCHMARK_MODULE = "asv_suite.comparisons"
COMPARISON_REPORT_DIRECTORY = "comparisons"
DOCUMENTATION_TIME_PLOT = "time_relative_to_gdal.svg"
DOCUMENTATION_MEMORY_PLOT = "peak_ram_by_raster_size.svg"
DOCUMENTATION_DATA = "benchmark_snapshot.json"
PERFORMANCE_CHANGE_REPORT = "performance-change.md"

# Colors stay consistent between the detailed ASV plots and the concise documentation snapshot
SERIES_COLORS = {
    "Eager": "#0072B2",
    "Dask": "#E69F00",
    "Multiprocessing": "#009E73",
    GDAL_CLI_LABEL: "#6C6C6C",
}


@dataclass(frozen=True)
class ComparisonMeasurement:
    """Store one fully identified benchmark result at one numeric parameter value."""

    comparison: str
    series_label: str
    operation: OperationName
    method: str | None
    calculation_engine: CalculationEngine | None
    strategy: OperationStrategyName | None
    execution_mode: ExecutionMode | None
    external_reference: ExternalReference | None
    series_dimension: ComparisonDimension
    parameter: int
    operation_time_s: float
    end_to_end_time_s: float
    peak_process_tree_mem_mb: float


def _benchmark_key(class_name: str, method_name: str) -> str:
    """Return the stable ASV result key for one comparison method."""

    # ASV names benchmarks relative to the configured benchmark directory
    return f"{COMPARISON_BENCHMARK_MODULE}.{class_name}.{method_name}"


def _latest_timestamp(result: Any) -> int:
    """Return the most recent measurement timestamp stored in one ASV result."""

    # Start times distinguish reruns of the same commit better than commit dates
    started_at = list(result.started_at.values())
    if started_at:
        return int(max(started_at))
    return int(result.date or 0)


def _required_benchmark_keys(comparisons: Iterable[Comparison] = COMPARISONS) -> set[str]:
    """Return every saved result needed to render the selected comparisons."""

    # A partially interrupted ASV run must not replace the latest complete documentation result
    methods = ("time_operation", "track_end_to_end_time_s", "track_peak_process_tree_mem_mb")
    return {
        _benchmark_key(class_name, method)
        for comparison in comparisons
        for _, class_name in comparison.series
        for method in methods
    }


def _select_complete_result(
    results: Iterable[Any],
    *,
    machine: str | None = None,
    commit: str | None = None,
    environment: str | None = None,
    require_complete: bool = True,
) -> Any:
    """Select the newest matching result from an ASV result sequence."""

    required_keys = _required_benchmark_keys()
    candidates = []
    for result in results:
        # Machine and commit filters make local multi-machine histories predictable
        if machine is not None and result.params.get("machine") != machine:
            continue
        if commit is not None and not result.commit_hash.startswith(commit):
            continue

        # ASV does not expose the environment name as a public property
        result_environment = str(getattr(result, "_env_name", ""))
        if environment is not None and result_environment != environment:
            continue
        if require_complete and not required_keys.issubset(result.get_all_result_keys()):
            continue
        candidates.append(result)

    if not candidates:
        raise RuntimeError("No matching saved ASV result contains the requested comparisons")
    return max(candidates, key=_latest_timestamp)


def select_asv_result(
    results_directory: Path,
    *,
    machine: str | None = None,
    commit: str | None = None,
    environment: str | None = None,
    require_complete: bool = True,
) -> Any:
    """Read ASV results and select the newest matching comparison result."""

    # Import ASV only when rendering so benchmark discovery remains lightweight
    from asv.results import iter_results

    return _select_complete_result(
        iter_results(str(results_directory)),
        machine=machine,
        commit=commit,
        environment=environment,
        require_complete=require_complete,
    )


def _decode_parameter(value: Any) -> int:
    """Convert the stable ASV parameter representation back to an integer."""

    # ASV stores parameters as repr strings even when benchmark inputs are numeric
    decoded = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(decoded, int):
        raise TypeError(f"Expected one integer ASV parameter, received {decoded!r}")
    return decoded


def _result_series(result: Any, key: str) -> tuple[list[int], list[float]]:
    """Return aligned numeric parameters and measurements for one ASV result key."""

    # Comparisons deliberately define exactly one parameter dimension
    params = result.get_result_params(key)
    if len(params) != 1:
        raise ValueError(f"Comparison must contain one parameter: {key}")
    parameters = [_decode_parameter(value) for value in params[0]]

    # Ask ASV to align values against the parameter order stored with this result
    values = result.get_result_value(key, params)
    if not isinstance(values, list):
        values = [values]
    if len(parameters) != len(values) or any(value is None for value in values):
        raise ValueError(f"Incomplete comparison result: {key}")
    return parameters, [float(value) for value in values]


def collect_comparison_measurements(
    result: Any,
    comparisons: Iterable[Comparison] = COMPARISONS,
) -> list[ComparisonMeasurement]:
    """Combine matching operation-time, end-to-end-time and peak-RAM results."""

    records = []
    available = set(result.get_all_result_keys())
    for comparison in comparisons:
        for series_label, class_name in comparison.series:
            benchmark_case = BENCHMARK_CASE_BY_CLASS.get(class_name)
            reference_case = EXTERNAL_REFERENCE_CASE_BY_CLASS.get(class_name)
            if (benchmark_case is None) == (reference_case is None):
                raise ValueError(f"Expected exactly one registered benchmark case for {class_name}")
            selected_case = benchmark_case or reference_case
            assert selected_case is not None
            if selected_case.operation != comparison.operation:
                raise ValueError(f"Comparison dimensions do not match registered case {class_name}")
            if comparison.series_dimension != "method" and selected_case.method != comparison.method:
                raise ValueError(f"Comparison method does not match registered case {class_name}")
            if benchmark_case is not None:
                if (
                    comparison.series_dimension != "calculation_engine"
                    and benchmark_case.calculation_engine != comparison.calculation_engine
                ):
                    raise ValueError(f"Comparison engine does not match registered case {class_name}")
                if (
                    comparison.series_dimension != "execution_mode"
                    and benchmark_case.execution_mode != comparison.execution_mode
                ):
                    raise ValueError(f"Comparison execution mode does not match registered case {class_name}")
                expected_strategy = comparison.strategy if benchmark_case.execution_mode != "eager" else None
                if comparison.series_dimension != "strategy" and benchmark_case.strategy != expected_strategy:
                    raise ValueError(f"Comparison strategy does not match registered case {class_name}")

            # The three ASV methods share the same numeric parameter values
            keys = {
                "operation": _benchmark_key(class_name, "time_operation"),
                "end_to_end": _benchmark_key(class_name, "track_end_to_end_time_s"),
                "memory": _benchmark_key(class_name, "track_peak_process_tree_mem_mb"),
            }
            missing = set(keys.values()) - available
            if missing:
                raise ValueError(f"Missing comparison results: {', '.join(sorted(missing))}")

            operation_params, operation_times = _result_series(result, keys["operation"])
            end_to_end_params, end_to_end_times = _result_series(result, keys["end_to_end"])
            memory_params, peak_memory = _result_series(result, keys["memory"])
            if operation_params != end_to_end_params or operation_params != memory_params:
                raise ValueError(f"Time and RAM parameters differ for {class_name}")

            # ASV stores skipped parameter combinations as NaN, which should leave no plotted metric
            records.extend(
                ComparisonMeasurement(
                    comparison=comparison.slug,
                    series_label=series_label,
                    operation=selected_case.operation,
                    method=selected_case.method,
                    calculation_engine=(benchmark_case.calculation_engine if benchmark_case is not None else None),
                    strategy=benchmark_case.strategy if benchmark_case is not None else None,
                    execution_mode=benchmark_case.execution_mode if benchmark_case is not None else None,
                    external_reference=(reference_case.external_reference if reference_case is not None else None),
                    series_dimension=comparison.series_dimension,
                    parameter=parameter,
                    operation_time_s=operation_time,
                    end_to_end_time_s=end_to_end_time,
                    peak_process_tree_mem_mb=memory,
                )
                for parameter, operation_time, end_to_end_time, memory in zip(
                    operation_params,
                    operation_times,
                    end_to_end_times,
                    peak_memory,
                )
                if all(math.isfinite(value) for value in (operation_time, end_to_end_time, memory))
            )
    return records


def _plot_comparison(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Write aligned operation-time, end-to-end-time and peak-RAM plots."""

    # Keep Matplotlib caches in a writable disposable location under sandboxed runs
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geoutils-matplotlib-cache"))

    # Import plotting only for explicit rendering commands
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout="constrained")
    for series_label, _ in comparison.series:
        # Stable parameter sorting keeps lines readable if ASV changes storage order
        selected = sorted(
            (
                record
                for record in records
                if record.comparison == comparison.slug and record.series_label == series_label
            ),
            key=lambda record: record.parameter,
        )
        parameters = [record.parameter for record in selected]
        axes[0].plot(parameters, [record.operation_time_s for record in selected], marker="o", label=series_label)
        axes[1].plot(parameters, [record.end_to_end_time_s for record in selected], marker="o", label=series_label)
        axes[2].plot(
            parameters,
            [record.peak_process_tree_mem_mb for record in selected],
            marker="o",
            label=series_label,
        )

    # Separate panels keep initialized work, complete execution and memory unambiguous
    axes[0].set_title(comparison.title)
    axes[0].set_ylabel("Operation time (s)")
    axes[1].set_ylabel("End-to-end time (s)")
    axes[2].set_ylabel("Peak process-tree memory (MB)")
    axes[2].set_xlabel(comparison.parameter_label)
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    if comparison.logarithmic_x:
        axes[2].set_xscale("log", base=2)

    # SVG stays sharp in local reports and on the published benchmark website
    figure.savefig(output, format="svg")
    plt.close(figure)


def _gdal_comparisons() -> tuple[Comparison, ...]:
    """Return raster-size comparisons that provide a GDAL CLI reference line."""

    return tuple(
        comparison
        for comparison in COMPARISONS
        if comparison.documentation and any(label == GDAL_CLI_LABEL for label, _ in comparison.series)
    )


def _largest_shared_parameter(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
) -> int:
    """Return the largest parameter measured by every series in one comparison."""

    # Shared parameters keep the normalized time bars based on exactly the same input size
    parameters_by_series = []
    for series_label, _ in comparison.series:
        parameters_by_series.append(
            {
                record.parameter
                for record in records
                if record.comparison == comparison.slug and record.series_label == series_label
            }
        )
    shared_parameters = set.intersection(*parameters_by_series)
    if not shared_parameters:
        raise ValueError(f"No shared parameter found for {comparison.slug}")
    return max(shared_parameters)


def _measurement_at(
    records: list[ComparisonMeasurement],
    comparison: str,
    series_label: str,
    parameter: int,
) -> ComparisonMeasurement:
    """Return one uniquely identified plot-series measurement."""

    selected = [
        record
        for record in records
        if record.comparison == comparison and record.series_label == series_label and record.parameter == parameter
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {series_label} result for {comparison} at parameter {parameter}, found {len(selected)}"
        )
    return selected[0]


def _documentation_operation_name(comparison: Comparison) -> str:
    """Return the concise operation name used on documentation graphics."""

    operation_labels = {
        "reproject": "Reprojection",
        "polygonize": "Polygonization",
        "rasterize": "Rasterization",
        "grid": "Nearest gridding",
    }
    return operation_labels[comparison.operation]


def _shared_change_parameter(
    comparison: Comparison,
    baseline_records: list[ComparisonMeasurement],
    current_records: list[ComparisonMeasurement],
) -> int | None:
    """Return the largest parameter shared by every requested series in both results."""

    series_labels = ("Eager", "Dask", "Multiprocessing", GDAL_CLI_LABEL)
    parameters = []
    for records in (baseline_records, current_records):
        for series_label in series_labels:
            parameters.append(
                {
                    record.parameter
                    for record in records
                    if record.comparison == comparison.slug and record.series_label == series_label
                }
            )
    if any(not values for values in parameters):
        return None
    shared = set.intersection(*parameters)
    return max(shared) if shared else None


def _format_change(value: float) -> str:
    """Describe one normalized performance change with an explicit direction."""

    if value >= 1:
        return f"{value:.2f}× faster"
    return f"{1 / value:.2f}× slower"


def performance_change_markdown(baseline_result: Any, current_result: Any) -> str:
    """Compare execution-mode time before and after a change, normalized to the GDAL CLI."""

    baseline_machine = str(baseline_result.params.get("machine", "unknown"))
    current_machine = str(current_result.params.get("machine", "unknown"))
    if baseline_machine != current_machine:
        raise ValueError(f"Performance results use different machines: {baseline_machine} and {current_machine}")

    # Normalizing each revision to its own GDAL CLI result limits machine-wide timing drift
    baseline_available = set(baseline_result.get_all_result_keys())
    current_available = set(current_result.get_all_result_keys())
    shared_comparisons = []
    baseline_records = []
    current_records = []
    for comparison in _gdal_comparisons():
        if not _required_benchmark_keys((comparison,)).issubset(baseline_available & current_available):
            continue
        try:
            baseline_comparison = collect_comparison_measurements(baseline_result, (comparison,))
            current_comparison = collect_comparison_measurements(current_result, (comparison,))
        except ValueError:
            # Interrupted runs may contain keys whose parameter series still has missing values
            continue
        shared_comparisons.append(comparison)
        baseline_records.extend(baseline_comparison)
        current_records.extend(current_comparison)
    execution_modes = ("Eager", "Dask", "Multiprocessing")
    ratios: dict[str, list[tuple[Comparison, int, float, float]]] = {name: [] for name in execution_modes}
    for comparison in shared_comparisons:
        parameter = _shared_change_parameter(comparison, baseline_records, current_records)
        if parameter is None:
            continue

        baseline_gdal = _measurement_at(baseline_records, comparison.slug, GDAL_CLI_LABEL, parameter).end_to_end_time_s
        current_gdal = _measurement_at(current_records, comparison.slug, GDAL_CLI_LABEL, parameter).end_to_end_time_s
        for execution_mode in execution_modes:
            baseline_ratio = (
                _measurement_at(baseline_records, comparison.slug, execution_mode, parameter).end_to_end_time_s
                / baseline_gdal
            )
            current_ratio = (
                _measurement_at(current_records, comparison.slug, execution_mode, parameter).end_to_end_time_s
                / current_gdal
            )
            ratios[execution_mode].append((comparison, parameter, baseline_ratio, current_ratio))

    if not all(ratios.values()):
        raise ValueError("No GDAL CLI comparison is complete in both performance results")

    # Geometric means combine multiplicative time ratios without favoring slower operations
    summary_rows = []
    for execution_mode in execution_modes:
        execution_mode_ratios = ratios[execution_mode]
        count = len(execution_mode_ratios)
        baseline_mean = math.prod(row[2] for row in execution_mode_ratios) ** (1 / count)
        current_mean = math.prod(row[3] for row in execution_mode_ratios) ** (1 / count)
        summary_rows.append((execution_mode, baseline_mean, current_mean, baseline_mean / current_mean))

    shared_rows = ratios[execution_modes[0]]
    operation_names = ", ".join(_documentation_operation_name(row[0]) for row in shared_rows)
    lines = [
        "# Performance change relative to the GDAL CLI",
        "",
        f"Baseline `{baseline_result.commit_hash}` → current `{current_result.commit_hash}` on `{current_machine}`.",
        "",
        "End-to-end time is normalized to the GDAL CLI for the same revision and input. Lower ratios are better. "
        "The summary "
        f"is the geometric mean across {operation_names}, using the largest parameter shared by every execution mode.",
        "",
        "| GeoUtils execution mode | Before / GDAL CLI | After / GDAL CLI | Change |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {execution_mode} | {baseline:.2f}× | {current:.2f}× | {_format_change(change)} |"
        for execution_mode, baseline, current, change in summary_rows
    )
    lines.extend(
        [
            "",
            "## Per-operation values",
            "",
            "| Operation | Input | Execution mode | Before / GDAL CLI | After / GDAL CLI | Change |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for execution_mode in execution_modes:
        for comparison, parameter, baseline, current in ratios[execution_mode]:
            lines.append(
                f"| {_documentation_operation_name(comparison)} | {parameter} | {execution_mode} | "
                f"{baseline:.2f}× | {current:.2f}× | {_format_change(baseline / current)} |"
            )
    return "\n".join(lines) + "\n"


def _plot_time_relative_to_gdal(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare end-to-end execution-mode time with the GDAL CLI on the largest shared raster."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    execution_modes = ("Eager", "Dask", "Multiprocessing")
    bar_width = 0.24
    positions = list(range(len(comparisons)))
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")

    for execution_mode_index, execution_mode in enumerate(execution_modes):
        ratios = []
        for comparison in comparisons:
            parameter = _largest_shared_parameter(comparison, records)
            gdal_time = _measurement_at(records, comparison.slug, GDAL_CLI_LABEL, parameter).end_to_end_time_s
            execution_mode_time = _measurement_at(
                records,
                comparison.slug,
                execution_mode,
                parameter,
            ).end_to_end_time_s
            ratios.append(execution_mode_time / gdal_time)

        # Group execution modes around each operation so their GDAL CLI ratio is easy to compare
        offset = (execution_mode_index - 1) * bar_width
        bars = axis.bar(
            [position + offset for position in positions],
            ratios,
            width=bar_width,
            label=execution_mode,
            color=SERIES_COLORS[execution_mode],
        )
        axis.bar_label(bars, fmt="%.1f×", padding=3, fontsize=8)

    # The GDAL CLI equals one by definition and remains visible when every execution mode is slower
    axis.axhline(
        1.0,
        color=SERIES_COLORS[GDAL_CLI_LABEL],
        linestyle="--",
        linewidth=1.2,
        label=GDAL_CLI_LABEL,
    )
    axis.set_xticks(positions, [_documentation_operation_name(comparison) for comparison in comparisons])
    axis.set_ylabel("End-to-end time relative to GDAL CLI")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=4)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_peak_ram_by_raster_size(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare total process-tree RAM as the raster size increases."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for axis, comparison in zip(axes.flat, comparisons):
        for series_label, _ in comparison.series:
            # Sorting protects the displayed dependency from ASV storage order
            selected = sorted(
                (
                    record
                    for record in records
                    if record.comparison == comparison.slug and record.series_label == series_label
                ),
                key=lambda record: record.parameter,
            )
            axis.plot(
                [record.parameter for record in selected],
                [record.peak_process_tree_mem_mb for record in selected],
                marker="o",
                label=series_label,
                color=SERIES_COLORS[series_label],
            )

        axis.set_title(_documentation_operation_name(comparison))
        axis.set_xlabel("Raster width and height (pixels)")
        axis.set_ylabel("Peak process-tree memory (MB)")
        axis.grid(alpha=0.3)

    # One legend applies to every panel and leaves the operation curves uncluttered
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def _result_payload(result: Any, records: list[ComparisonMeasurement]) -> dict[str, Any]:
    """Return measurements and enough run metadata to interpret them later."""

    return {
        "metadata": {
            "commit": result.commit_hash,
            "date": result.date,
            "environment": str(getattr(result, "_env_name", "")),
            "machine": result.params,
            "operation_time": "wall-clock time after runner initialization",
            "end_to_end_time": "wall-clock time including runner initialization but excluding input generation",
            "memory": "peak aggregate process-tree memory",
        },
        "measurements": [asdict(record) for record in records],
    }


def render_documentation_snapshot(
    result: Any,
    output_directory: Path,
) -> list[ComparisonMeasurement]:
    """Write the two documentation graphics and their complete numeric source."""

    # Keep Matplotlib caches in a writable disposable location under sandboxed runs
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geoutils-matplotlib-cache"))
    output_directory.mkdir(parents=True, exist_ok=True)
    records = collect_comparison_measurements(result)

    # Documentation uses two broad summaries while the ASV site retains every detailed panel
    _plot_time_relative_to_gdal(records, output_directory / DOCUMENTATION_TIME_PLOT)
    _plot_peak_ram_by_raster_size(records, output_directory / DOCUMENTATION_MEMORY_PLOT)
    (output_directory / DOCUMENTATION_DATA).write_text(
        json.dumps(_result_payload(result, records), indent=2) + "\n",
        encoding="utf-8",
    )
    return records


def _markdown_report(result: Any, *, includes_performance_change: bool = False) -> str:
    """Return a compact Markdown index linking plots and numeric results."""

    machine = result.params.get("machine", "unknown")
    lines = [
        "# GeoUtils comparisons",
        "",
        f"Commit `{result.commit_hash}` on machine `{machine}`",
        "",
        "Operation time excludes runner initialization. End-to-end time includes it but excludes input generation. "
        "Peak memory is aggregated over the benchmark process and all runner children.",
        "",
        "Raw values: [CSV](comparisons.csv) · [JSON](comparisons.json)",
    ]
    if includes_performance_change:
        lines.extend(["", f"Before/after summary: [{PERFORMANCE_CHANGE_REPORT}]({PERFORMANCE_CHANGE_REPORT})"])
    for comparison in COMPARISONS:
        lines.extend(["", f"## {comparison.title}", "", f"![{comparison.title}]({comparison.slug}.svg)"])
    return "\n".join(lines) + "\n"


def _comparison_html(result: Any, *, includes_performance_change: bool = False) -> str:
    """Return the standalone comparison browser page."""

    machine = html.escape(str(result.params.get("machine", "unknown")))
    commit = html.escape(str(result.commit_hash))
    sections = []
    for comparison in COMPARISONS:
        title = html.escape(comparison.title)
        sections.append(f'<section><h2>{title}</h2><img src="{comparison.slug}.svg" alt="{title}"></section>')
    change_link = (
        f'<p>Before/after summary: <a href="{PERFORMANCE_CHANGE_REPORT}">{PERFORMANCE_CHANGE_REPORT}</a></p>'
        if includes_performance_change
        else ""
    )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>GeoUtils comparisons</title>",
            "<style>body{font-family:sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem}",
            "img{width:100%;height:auto}code{overflow-wrap:anywhere}</style></head><body>",
            '<p><a href="../index.html">GeoUtils benchmark results</a></p>',
            "<h1>Comparisons</h1>",
            f"<p>Commit <code>{commit}</code> on machine <code>{machine}</code></p>",
            "<p>Operation time excludes runner initialization. End-to-end time includes it but excludes input "
            "generation. Peak memory is aggregated over the benchmark process and all runner children.</p>",
            '<p>Raw values: <a href="comparisons.csv">CSV</a> · ' '<a href="comparisons.json">JSON</a></p>',
            change_link,
            *sections,
            "</body></html>",
        ]
    )


def _site_index() -> str:
    """Return the single landing page for every generated benchmark view."""

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>GeoUtils benchmark results</title>",
            "<style>body{font-family:sans-serif;max-width:760px;margin:3rem auto;padding:0 1rem}",
            "li{margin:1rem 0}</style></head><body>",
            "<h1>GeoUtils benchmark results</h1>",
            "<p>Choose the view that matches the question being investigated.</p>",
            "<ul>",
            '<li><a href="asv/index.html">Performance across commits</a> — native ASV history</li>',
            '<li><a href="comparisons/index.html">Comparisons</a> — '
            "operation time, end-to-end time and peak RAM</li>",
            "</ul>",
            "</body></html>",
        ]
    )


def render_comparisons(
    result: Any,
    website_directory: Path,
    *,
    baseline_result: Any | None = None,
) -> list[ComparisonMeasurement]:
    """Write comparison plots and one landing page beside the native ASV site."""

    # All generated material stays under the gitignored ASV website directory
    report_directory = website_directory / COMPARISON_REPORT_DIRECTORY
    report_directory.mkdir(parents=True, exist_ok=True)
    records = collect_comparison_measurements(result)
    for comparison in COMPARISONS:
        _plot_comparison(comparison, records, report_directory / f"{comparison.slug}.svg")

    # JSON retains run metadata while CSV remains convenient for independent analysis
    payload = _result_payload(result, records)
    (report_directory / "comparisons.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    # Write explicit columns so the stable schema does not depend on plotting code
    with open(report_directory / "comparisons.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    includes_performance_change = baseline_result is not None
    if baseline_result is not None:
        (report_directory / PERFORMANCE_CHANGE_REPORT).write_text(
            performance_change_markdown(baseline_result, result), encoding="utf-8"
        )
    (report_directory / "README.md").write_text(
        _markdown_report(result, includes_performance_change=includes_performance_change), encoding="utf-8"
    )
    (report_directory / "index.html").write_text(
        _comparison_html(result, includes_performance_change=includes_performance_change), encoding="utf-8"
    )

    # A small stable root avoids modifying files generated internally by ASV
    website_directory.mkdir(parents=True, exist_ok=True)
    (website_directory / "index.html").write_text(_site_index(), encoding="utf-8")
    return records


def main() -> None:
    """Select one saved ASV result and render its requested report outputs."""

    # This command runs explicitly after asv publish and is not called during ASV measurement
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=Path("benchmarks/results/asv/results"))
    parser.add_argument("--website-directory", type=Path, default=Path("benchmarks/results/asv/html"))
    parser.add_argument("--doc-dir", type=Path)
    parser.add_argument("--doc-only", action="store_true")
    parser.add_argument("--machine")
    parser.add_argument("--commit")
    parser.add_argument("--baseline-commit")
    parser.add_argument("--environment")
    args = parser.parse_args()

    # Optional filters select a reproducible record from histories with several machines
    result = select_asv_result(
        args.results_directory,
        machine=args.machine,
        commit=args.commit,
        environment=args.environment,
    )
    baseline_result = None
    if args.baseline_commit is not None:
        baseline_result = select_asv_result(
            args.results_directory,
            machine=str(result.params.get("machine", "")),
            commit=args.baseline_commit,
            environment=str(getattr(result, "_env_name", "")),
            require_complete=False,
        )
    if not args.doc_only:
        render_comparisons(result, args.website_directory, baseline_result=baseline_result)
    if args.doc_dir is not None:
        render_documentation_snapshot(result, args.doc_dir)
    elif args.doc_only:
        parser.error("--doc-only requires --doc-dir")


if __name__ == "__main__":
    main()
