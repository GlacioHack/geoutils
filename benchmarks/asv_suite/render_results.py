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
from typing import Any, Literal

from benchmarks.asv_suite.comparisons import (
    BENCHMARK_CASE_BY_CLASS,
    CALCULATION_ENGINE_LABELS,
    COMPARISONS,
    EXTERNAL_REFERENCE_CASE_BY_CLASS,
    GDAL_CLI_LABEL,
    METHOD_LABELS,
    STRATEGY_LABELS,
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

# Name the files written by this renderer and linked from the benchmark website and documentation
COMPARISON_BENCHMARK_MODULE = "asv_suite.comparisons"
COMPARISON_REPORT_DIRECTORY = "comparisons"
SCALING_REPORT_PAGE = "scaling.html"
HISTORY_REPORT_PAGE = "history.html"
DOCUMENTATION_TIME_PLOT = "time_relative_to_gdal.svg"
DOCUMENTATION_MEMORY_PLOT = "peak_ram_by_raster_size.svg"
DOCUMENTATION_DATA = "benchmark_snapshot.json"
PERFORMANCE_CHANGE_REPORT = "performance-change.md"
PREVIEW_WEBSITE_DIRECTORY = Path("benchmarks/results/asv/preview")

# Colors stay consistent between the detailed ASV plots and the concise documentation snapshot
SERIES_COLORS = {
    "Eager": "#0072B2",
    "Dask": "#E69F00",
    "Multiprocessing": "#009E73",
    GDAL_CLI_LABEL: "#6C6C6C",
}

# Group plots by the GeoUtils choice represented by their separate lines; every plot still scales a numeric input
COMPARISON_SECTION_DETAILS: dict[ComparisonDimension, tuple[str, str]] = {
    "execution_mode": (
        "Execution modes",
        "Compare Eager, Dask and Multiprocessing with one worker, showing coordination costs without parallel "
        "speed-up.",
    ),
    "calculation_engine": (
        "Calculation engines",
        "Compare numerical libraries during eager execution, so scheduling remains fixed.",
    ),
    "method": (
        "Operation methods",
        "Compare operation-specific algorithms, such as nearest, linear/Delaunay and inverse-distance gridding.",
    ),
    "strategy": (
        "Chunk strategies",
        "Compare how chunked operations reconcile partial results at chunk boundaries.",
    ),
}

# Scaling sections answer which input grows, independently of the GeoUtils choices represented by plot lines
SCALING_SECTION_DETAILS = {
    "Size of raster (pixels per side)": (
        "Scaling with raster size",
        "Raster width and height vary together; other inputs remain fixed.",
    ),
    "Size of chunks (pixels per side)": (
        "Scaling with chunk size",
        "Chunk width and height vary while the complete raster remains fixed.",
    ),
    "Number of interpolated points": (
        "Scaling with the number of interpolated points",
        "The number of requested point locations varies while the source raster remains fixed.",
    ),
    "Number of source points per axis": (
        "Scaling with the number of source points",
        "The number of source points varies while the output raster remains fixed.",
    ),
    "Number of sampled values": (
        "Scaling with the number of sampled values",
        "The returned sample size varies while the source raster remains fixed.",
    ),
}

OPERATION_LABELS: dict[OperationName, str] = {
    "crop": "Cropping",
    "translate": "Translation",
    "copy": "Copying",
    "filter": "Filtering",
    "reproject": "Reprojection",
    "statistics": "Statistics",
    "subsample": "Subsampling",
    "interp_points": "Point interpolation",
    "polygonize": "Polygonization",
    "write": "Writing",
    "rasterize": "Rasterization",
    "create_mask": "Mask creation",
    "grid": "Gridding",
}

# Group operations by input and output type, using the same data-flow labels as the scalability documentation
OPERATION_GROUP_ORDER = (
    "Raster ⟶ Raster",
    "Point ⟶ Point",
    "Raster ⟶ Point",
    "Raster ⟶ Vector",
    "Point ⟶ Raster",
    "Vector ⟶ Raster",
    "Raster ⟶ Other",
)
OPERATION_GROUPS: dict[OperationName, str] = {
    "crop": "Raster ⟶ Raster",
    "translate": "Raster ⟶ Raster",
    "copy": "Raster ⟶ Raster",
    "filter": "Raster ⟶ Raster",
    "reproject": "Raster ⟶ Raster",
    "subsample": "Raster ⟶ Point",
    "interp_points": "Raster ⟶ Point",
    "polygonize": "Raster ⟶ Vector",
    "grid": "Point ⟶ Raster",
    "rasterize": "Vector ⟶ Raster",
    "create_mask": "Vector ⟶ Raster",
    "statistics": "Raster ⟶ Other",
    "write": "Raster ⟶ Other",
}


# Store one row per plot point, with operation, method, engine, strategy and execution mode in separate fields
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


class _PreviewResult:
    """Provide complete deterministic ASV-like measurements for renderer development."""

    def __init__(self, commit_hash: str = "preview-current", geoutils_scale: float = 1.0) -> None:
        """Create three sample values for every comparison series and measurement."""

        self.commit_hash = commit_hash
        self.date = 1_700_000_000_000
        self.params = {"machine": "preview-machine", "cpu": "preview-cpu"}
        self.started_at = {"benchmark": 1_700_000_001_000}
        self._env_name = "preview"
        self.values: dict[str, list[float]] = {}
        self.parameters: dict[str, list[list[str]]] = {}

        # Create all keys normally read from saved ASV results so every report section can render
        for comparison in COMPARISONS:
            if comparison.parameter_label == "Number of interpolated points":
                parameter_values = (256, 2048, 16384)
            elif comparison.parameter_label == "Number of source points per axis":
                parameter_values = (3, 9, 33)
            elif comparison.parameter_label == "Number of sampled values":
                parameter_values = (256, 2048, 16384)
            elif comparison.parameter_label == "Size of chunks (pixels per side)":
                parameter_values = (256, 512, 1024)
            elif comparison.operation == "grid":
                parameter_values = (512, 1024, 2048)
            else:
                parameter_values = (1024, 2048, 4096)

            for series_index, (_, class_name) in enumerate(comparison.series, start=1):
                prefix = f"{COMPARISON_BENCHMARK_MODULE}.{class_name}"
                if f"{prefix}.time_operation" in self.values:
                    continue

                # Scale GeoUtils values between fake revisions while leaving the GDAL reference stable
                scale = geoutils_scale if class_name in BENCHMARK_CASE_BY_CLASS else 1.0
                parameters = [[repr(value) for value in parameter_values]]
                self.parameters[f"{prefix}.time_operation"] = parameters
                self.parameters[f"{prefix}.track_end_to_end_time_s"] = parameters
                self.parameters[f"{prefix}.track_peak_process_tree_mem_mb"] = parameters
                self.values[f"{prefix}.time_operation"] = [scale * series_index * value for value in (0.05, 0.10, 0.20)]
                self.values[f"{prefix}.track_end_to_end_time_s"] = [
                    scale * series_index * value for value in (0.10, 0.20, 0.40)
                ]
                self.values[f"{prefix}.track_peak_process_tree_mem_mb"] = [
                    scale * series_index * value for value in (100.0, 140.0, 200.0)
                ]

    def get_all_result_keys(self) -> Any:
        """Return every generated benchmark result key."""

        return self.values.keys()

    def get_result_params(self, key: str) -> list[list[str]]:
        """Return the numeric parameter axis stored for one benchmark."""

        return self.parameters[key]

    def get_result_value(self, key: str, params: list[list[str]]) -> list[float]:
        """Return measurements aligned with the requested parameter axis."""

        if params != self.parameters[key]:
            raise ValueError(f"Preview parameters do not match the generated values for {key}")
        return self.values[key]


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
    """Combine matching operation-only, elapsed-time and peak-memory results."""

    # Convert each saved ASV series into one record per numeric input value
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
    """Write aligned operation-only, elapsed-time and peak-memory plots."""

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
    axes[0].set_ylabel("Operation only (s)")
    axes[1].set_ylabel("Elapsed time (s)")
    axes[2].set_ylabel("Peak combined memory (MB)")
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

    # Convert registry names such as "reproject" to the titles shown in public plots and tables
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

    # Compare only input sizes measured for all three GeoUtils execution modes and the GDAL CLI on both revisions
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

    # Build the report only from operations with all three measurements on both revisions
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

    # Group GDAL-normalized timings by execution mode before summarizing across operations
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
        "Elapsed time is normalized to the GDAL CLI for the same revision and input. Lower ratios are better. "
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
    """Compare elapsed execution-mode time with the GDAL CLI on the largest shared raster."""

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
    axis.set_ylabel("Elapsed time relative to GDAL CLI")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=4)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_peak_ram_by_raster_size(
    records: list[ComparisonMeasurement],
    output: Path,
) -> None:
    """Compare peak combined memory as the raster size increases."""

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
        axis.set_xlabel("Size of raster (pixels per side)")
        axis.set_ylabel("Peak combined memory (MB)")
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
            "operation_time": "wall-clock time after execution-mode initialization",
            "end_to_end_time": "elapsed wall-clock time from prepared inputs through completed output",
            "memory": "peak combined memory of the benchmark process and its child workers",
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


def _operation_and_method_label(comparison: Comparison) -> str:
    """Return a concise operation label including its fixed method or strategy."""

    label = OPERATION_LABELS[comparison.operation]
    if comparison.method is not None:
        method = (
            METHOD_LABELS.get(comparison.method, comparison.method.replace("_", " ").title())
            if comparison.operation == "grid"
            else comparison.method.replace("_", " ").title()
        )
        label = f"{label} — {method}"
    if comparison.strategy is not None:
        label = f"{label} — {STRATEGY_LABELS[comparison.strategy]}"
    return label


def _workload_label(comparison: Comparison, parameter: int) -> str:
    """Describe every input size that identifies one representative benchmark workload."""

    return comparison.workload_template.format(parameter=f"{parameter:,}")


def _workload_html(comparison: Comparison, parameter: int) -> str:
    """Place each input dimension on a separate line in an HTML workload cell."""

    parts = _workload_label(comparison, parameter).split("; ")
    return '<span class="workload-list">' + "".join(f"<span>{html.escape(part)}</span>" for part in parts) + "</span>"


def _largest_parameter_for_series(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    series_labels: Iterable[str],
) -> int:
    """Return the largest parameter measured for every selected plot series."""

    parameters = [
        {
            record.parameter
            for record in records
            if record.comparison == comparison.slug and record.series_label == series_label
        }
        for series_label in series_labels
    ]
    if not parameters or any(not values for values in parameters):
        raise ValueError(f"No complete parameter values found for {comparison.slug}")
    shared = set.intersection(*parameters)
    if not shared:
        raise ValueError(f"No shared parameter found for {comparison.slug}")
    return max(shared)


def _group_operation_rows(rows: list[tuple[Comparison, str]], *, column_count: int) -> str:
    """Group operation table rows by their input and output data types."""

    bodies = []
    rendered_rows = 0
    for group in OPERATION_GROUP_ORDER:
        selected = [row for comparison, row in rows if OPERATION_GROUPS.get(comparison.operation) == group]
        if not selected:
            continue
        rendered_rows += len(selected)
        bodies.append(
            f'<tbody><tr class="operation-group"><th colspan="{column_count}" scope="rowgroup">{group}</th></tr>'
            f'{"".join(selected)}</tbody>'
        )
    if rendered_rows != len(rows):
        raise ValueError("Every benchmark operation must belong to one input/output group")
    return "".join(bodies)


def _format_measurement_cell(
    measurement: ComparisonMeasurement | None,
    *,
    reference: ComparisonMeasurement | None,
    reference_label: str,
    best_time: float | None,
    lowest_memory: float | None,
) -> str:
    """Render elapsed time and peak memory with optional relative values."""

    if measurement is None:
        return '<span class="unavailable" aria-label="Not supported">—</span>'
    time_class = " best" if best_time is not None and math.isclose(measurement.end_to_end_time_s, best_time) else ""
    memory_class = (
        " best"
        if lowest_memory is not None and math.isclose(measurement.peak_process_tree_mem_mb, lowest_memory)
        else ""
    )
    time_relative = ""
    memory_relative = ""
    if reference is not None:
        time_relative = (
            f'<span class="ratio">{measurement.end_to_end_time_s / reference.end_to_end_time_s:.2f}× '
            f"{reference_label}</span>"
        )
        memory_relative = (
            f'<span class="ratio">{measurement.peak_process_tree_mem_mb / reference.peak_process_tree_mem_mb:.2f}× '
            f"{reference_label}</span>"
        )
    return "".join(
        [
            f'<span class="metric{time_class}"><span class="metric-label">Time</span>'
            f"{measurement.end_to_end_time_s:.3g} s{time_relative}</span>",
            f'<span class="metric{memory_class}"><span class="metric-label">Memory</span>'
            f"{measurement.peak_process_tree_mem_mb:.3g} MB{memory_relative}</span>",
        ]
    )


def _eager_implementation_measurements(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    parameter: int | None = None,
) -> tuple[int, dict[CalculationEngine | ExternalReference, ComparisonMeasurement]]:
    """Return eager GeoUtils engines and any external GDAL result at one shared workload."""

    selected_labels = [
        label
        for label, _ in comparison.series
        if label == GDAL_CLI_LABEL or comparison.series_dimension == "calculation_engine" or label == "Eager"
    ]
    if parameter is None:
        parameter = _largest_parameter_for_series(comparison, records, selected_labels)
    selected = (
        record
        for record in records
        if record.comparison == comparison.slug
        and record.parameter == parameter
        and record.series_label in selected_labels
    )
    by_implementation: dict[CalculationEngine | ExternalReference, ComparisonMeasurement] = {}
    for measurement in selected:
        implementation = measurement.external_reference or measurement.calculation_engine
        if implementation is not None:
            by_implementation[implementation] = measurement
    return parameter, by_implementation


def _execution_mode_measurements(
    comparison: Comparison,
    records: list[ComparisonMeasurement],
    parameter: int | None = None,
) -> tuple[int, dict[ExecutionMode, ComparisonMeasurement]]:
    """Return GeoUtils execution modes at the largest workload shared by every mode."""

    selected_labels = [label for label, _ in comparison.series if label != GDAL_CLI_LABEL]
    if parameter is None:
        parameter = _largest_parameter_for_series(comparison, records, selected_labels)
    measurements = (
        record
        for record in records
        if record.comparison == comparison.slug
        and record.parameter == parameter
        and record.external_reference is None
        and record.execution_mode is not None
    )
    by_mode: dict[ExecutionMode, ComparisonMeasurement] = {}
    for measurement in measurements:
        assert measurement.execution_mode is not None
        by_mode[measurement.execution_mode] = measurement
    return parameter, by_mode


def _engine_summary_comparisons() -> tuple[Comparison, ...]:
    """Select one representative input axis for every eager operation and method."""

    # Prefer a direct engine comparison, then fall back to an execution comparison containing Eager
    selected: dict[tuple[OperationName, str | None], tuple[int, Comparison]] = {}
    for comparison in COMPARISONS:
        priority = 0
        if comparison.series_dimension == "calculation_engine":
            priority = 2 if comparison.parameter_label == "Size of raster (pixels per side)" else 1
        elif comparison.series_dimension == "execution_mode" and any(
            label == "Eager" for label, _ in comparison.series
        ):
            priority = 1
        if priority == 0:
            continue
        key = (comparison.operation, comparison.method)
        current = selected.get(key)
        if current is None or priority > current[0]:
            selected[key] = (priority, comparison)
    return tuple(item[1] for item in selected.values())


def _engine_summary_table(records: list[ComparisonMeasurement]) -> str:
    """Compare eager GeoUtils calculation engines with the external GDAL CLI."""

    rows: list[tuple[Comparison, str]] = []
    columns: tuple[CalculationEngine | ExternalReference, ...] = ("rasterio", "scipy", "numba", "gdal_cli")
    for comparison in _engine_summary_comparisons():
        parameter = _summary_reference_parameter(comparison, records)
        _, by_implementation = _eager_implementation_measurements(comparison, records, parameter)
        gdal = by_implementation.get("gdal_cli")
        best_time = min(measurement.end_to_end_time_s for measurement in by_implementation.values())
        lowest_memory = min(measurement.peak_process_tree_mem_mb for measurement in by_implementation.values())
        cells = [
            _format_measurement_cell(
                by_implementation.get(column),
                reference=gdal if column != "gdal_cli" else None,
                reference_label="GDAL CLI",
                best_time=best_time,
                lowest_memory=lowest_memory,
            )
            for column in columns
        ]
        rows.append(
            (
                comparison,
                "<tr>"
                f'<th scope="row">{html.escape(_operation_and_method_label(comparison))}</th>'
                f"<td>{_workload_html(comparison, parameter)}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>",
            )
        )
    return "".join(
        [
            '<div class="table-wrap"><table><thead>',
            '<tr><th class="row-heading" rowspan="2" scope="col">Operation and method</th>',
            '<th class="workload-heading" rowspan="2" scope="col">Reference workload</th>',
            '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils calculation engine</th>',
            '<th class="external-heading group-heading" scope="colgroup">External reference</th></tr>',
            '<tr><th scope="col">Rasterio/GDAL</th><th scope="col">SciPy</th><th scope="col">Numba</th>',
            '<th class="external-heading" scope="col">GDAL CLI</th></tr></thead>',
            _group_operation_rows(rows, column_count=6),
            "</table></div>",
        ]
    )


def _execution_summary_comparisons() -> tuple[Comparison, ...]:
    """Return every plot that directly compares GeoUtils execution modes."""

    return tuple(comparison for comparison in COMPARISONS if comparison.series_dimension == "execution_mode")


def _execution_summary_table(records: list[ComparisonMeasurement]) -> str:
    """Compare Eager, Dask and Multiprocessing with one fixed calculation engine."""

    rows: list[tuple[Comparison, str]] = []
    columns: tuple[ExecutionMode, ...] = ("eager", "dask", "multiprocessing")
    for comparison in _execution_summary_comparisons():
        parameter = _summary_reference_parameter(comparison, records)
        _, by_mode = _execution_mode_measurements(comparison, records, parameter)
        eager = by_mode.get("eager")
        best_time = min(measurement.end_to_end_time_s for measurement in by_mode.values())
        lowest_memory = min(measurement.peak_process_tree_mem_mb for measurement in by_mode.values())
        cells = [
            _format_measurement_cell(
                by_mode.get(column),
                reference=eager,
                reference_label="Eager",
                best_time=best_time,
                lowest_memory=lowest_memory,
            )
            for column in columns
        ]
        engine = (
            CALCULATION_ENGINE_LABELS[comparison.calculation_engine]
            if comparison.calculation_engine is not None
            else "No separate engine"
        )
        rows.append(
            (
                comparison,
                "<tr>"
                f'<th scope="row">{html.escape(_operation_and_method_label(comparison))}</th>'
                f"<td>{html.escape(engine)}</td>"
                f"<td>{_workload_html(comparison, parameter)}</td>"
                + "".join(f"<td>{cell}</td>" for cell in cells)
                + "</tr>",
            )
        )
    return "".join(
        [
            '<div class="table-wrap"><table><thead>',
            '<tr><th class="row-heading" rowspan="2" scope="col">Operation and method</th>',
            '<th rowspan="2" scope="col">Engine</th>',
            '<th class="workload-heading" rowspan="2" scope="col">Reference workload</th>',
            '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils execution mode</th></tr>',
            '<tr><th scope="col">Eager</th><th scope="col">Dask</th><th scope="col">Multiprocessing</th>',
            "</tr></thead>",
            _group_operation_rows(rows, column_count=6),
            "</table></div>",
        ]
    )


def _summary_reference_parameter(comparison: Comparison, records: list[ComparisonMeasurement]) -> int:
    """Use the common raster size for summary rows and the largest value on other input axes."""

    if comparison.parameter_label == "Size of raster (pixels per side)" and all(
        any(
            record.comparison == comparison.slug and record.series_label == series_label and record.parameter == 2048
            for record in records
        )
        for series_label, _ in comparison.series
    ):
        return 2048
    return _largest_shared_parameter(comparison, records)


def _headline_summary_table(records: list[ComparisonMeasurement]) -> str:
    """List engines and execution modes in separate columns beside the external GDAL CLI."""

    engine_order: tuple[CalculationEngine, ...] = ("rasterio", "scipy", "numba")
    mode_order: tuple[ExecutionMode, ...] = ("eager", "dask", "multiprocessing")
    engine_comparisons = {
        (comparison.operation, comparison.method): comparison for comparison in _engine_summary_comparisons()
    }
    execution_comparisons = {
        (comparison.operation, comparison.method): comparison for comparison in _execution_summary_comparisons()
    }
    comparison_keys = tuple(
        dict.fromkeys(
            (comparison.operation, comparison.method)
            for comparison in COMPARISONS
            if (comparison.operation, comparison.method) in engine_comparisons
            or (comparison.operation, comparison.method) in execution_comparisons
        )
    )
    rows: list[tuple[Comparison, str]] = []
    for comparison_key in comparison_keys:
        engine_comparison = engine_comparisons.get(comparison_key)
        execution_comparison = execution_comparisons.get(comparison_key)
        comparison = engine_comparison or execution_comparison
        assert comparison is not None
        parameter = _summary_reference_parameter(comparison, records)

        if engine_comparison is None:
            implementations: dict[CalculationEngine | ExternalReference, ComparisonMeasurement] = {}
        else:
            _, implementations = _eager_implementation_measurements(engine_comparison, records, parameter)

        if execution_comparison is None:
            modes: dict[ExecutionMode, ComparisonMeasurement] = {}
        else:
            _, modes = _execution_mode_measurements(execution_comparison, records, parameter)

        gdal = implementations.get("gdal_cli")
        engine_cells = [
            _format_measurement_cell(
                implementations.get(engine),
                reference=None,
                reference_label="",
                best_time=None,
                lowest_memory=None,
            )
            for engine in engine_order
        ]
        execution_cells = [
            _format_measurement_cell(
                modes.get(mode),
                reference=None,
                reference_label="",
                best_time=None,
                lowest_memory=None,
            )
            for mode in mode_order
        ]
        gdal_cell = _format_measurement_cell(
            gdal,
            reference=None,
            reference_label="",
            best_time=None,
            lowest_memory=None,
        )

        rows.append(
            (
                comparison,
                "<tr>"
                f'<th scope="row">{html.escape(_operation_and_method_label(comparison))}</th>'
                f"<td>{_workload_html(comparison, parameter)}</td>"
                + "".join(f"<td>{cell}</td>" for cell in (*engine_cells, *execution_cells))
                + f'<td class="external-cell">{gdal_cell}</td></tr>',
            )
        )
    return "".join(
        [
            '<div class="table-wrap"><table class="summary-table"><thead>',
            '<tr><th class="row-heading" rowspan="2" scope="col">Operation and method</th>',
            '<th class="workload-heading" rowspan="2" scope="col">Reference workload</th>',
            '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils calculation engine</th>',
            '<th class="group-heading" colspan="3" scope="colgroup">GeoUtils execution mode</th>',
            '<th class="external-heading group-heading" scope="colgroup">External reference</th></tr>',
            '<tr><th scope="col">Rasterio/GDAL</th><th scope="col">SciPy</th><th scope="col">Numba</th>',
            '<th scope="col">Eager</th><th scope="col">Dask</th><th scope="col">Multiprocessing</th>',
            '<th class="external-heading" scope="col">GDAL CLI</th></tr></thead>',
            _group_operation_rows(rows, column_count=9),
            "</table></div>",
        ]
    )


def _choice_tables(dimension: Literal["method", "strategy"], records: list[ComparisonMeasurement]) -> str:
    """Render representative time and memory for methods or chunk strategies."""

    articles = []
    choice_name = "method" if dimension == "method" else "chunk strategy"
    for comparison in (item for item in COMPARISONS if item.series_dimension == dimension):
        parameter = _summary_reference_parameter(comparison, records)
        measurements = [
            _measurement_at(records, comparison.slug, series_label, parameter) for series_label, _ in comparison.series
        ]
        headers = "".join(f'<th scope="col">{html.escape(label)}</th>' for label, _ in comparison.series)
        cells = "".join(
            "<td>"
            + _format_measurement_cell(
                measurement,
                reference=None,
                reference_label="",
                best_time=None,
                lowest_memory=None,
            )
            + "</td>"
            for measurement in measurements
        )
        articles.append(
            '<article class="choice-card">'
            f"<h3>{html.escape(comparison.title)}</h3><p>{html.escape(comparison.description)}</p>"
            f'<p class="workload"><strong>Reference workload:</strong> '
            f"{_workload_html(comparison, parameter)}</p>"
            '<div class="table-wrap"><table class="choice-table"><thead>'
            f'<tr><th colspan="{len(measurements)}" scope="colgroup">GeoUtils {choice_name}</th></tr>'
            f"<tr>{headers}</tr></thead><tbody><tr>{cells}</tr></tbody></table></div>"
            f'<p><a href="scaling.html#{html.escape(comparison.slug)}">View how this comparison scales</a></p>'
            "</article>"
        )
    return "".join(articles)


def _markdown_report(result: Any, *, includes_performance_change: bool = False) -> str:
    """Return a compact Markdown index linking plots and numeric results."""

    machine = result.params.get("machine", "unknown")
    lines = [
        "# GeoUtils comparisons",
        "",
        f"Commit `{result.commit_hash}` on machine `{machine}`",
        "",
        "Operation-only time starts after execution-mode initialization. Elapsed time runs from prepared inputs "
        "through completed output and is the comparable boundary for GeoUtils versus GDAL CLI. Peak memory combines "
        "the benchmark process and all child workers.",
        "",
        "Raw values: [CSV](comparisons.csv) · [JSON](comparisons.json)",
    ]
    if includes_performance_change:
        lines.extend(["", f"Before/after summary: [{PERFORMANCE_CHANGE_REPORT}]({PERFORMANCE_CHANGE_REPORT})"])
    for dimension, (section_title, section_description) in COMPARISON_SECTION_DETAILS.items():
        comparisons = tuple(comparison for comparison in COMPARISONS if comparison.series_dimension == dimension)
        if not comparisons:
            continue
        lines.extend(["", f"## {section_title}", "", section_description])
        for comparison in comparisons:
            lines.extend(
                [
                    "",
                    f"### {comparison.title}",
                    "",
                    comparison.description,
                    "",
                    f"![{comparison.title}]({comparison.slug}.svg)",
                ]
            )
    return "\n".join(lines) + "\n"


_SITE_STYLE = """
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #5b6475;
  --line: #dce2ea;
  --surface: #ffffff;
  --soft: #f5f7fb;
  --blue: #1463df;
  --blue-soft: #eaf2ff;
  --teal: #087f74;
  --teal-soft: #e6f7f4;
  --purple: #7452b8;
  --purple-soft: #f1edfa;
  --amber: #a46008;
  --amber-soft: #fff3df;
  --external: #5c6675;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--surface);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
}
a { color: #075dcc; text-underline-offset: .16em; }
.site-header { position: sticky; top: 0; z-index: 20; border-bottom: 1px solid var(--line); background: #fffffff2; backdrop-filter: blur(10px); }
.header-inner { display: flex; align-items: center; justify-content: space-between; gap: 1rem; max-width: 1440px; margin: 0 auto; padding: .75rem 1.25rem; }
.brand { color: var(--ink); font-weight: 750; text-decoration: none; white-space: nowrap; }
.site-nav { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .25rem; }
.site-nav a { color: var(--muted); padding: .45rem .7rem; border-radius: .45rem; font-size: .92rem; text-decoration: none; }
.site-nav a:hover, .site-nav a[aria-current="page"] { color: var(--ink); background: var(--blue-soft); }
main { max-width: 1400px; margin: 0 auto; padding: 2.5rem 1.25rem 5rem; }
.hero { max-width: 960px; margin-bottom: 2rem; }
.eyebrow { color: var(--blue); font-size: .78rem; font-weight: 750; letter-spacing: .09em; text-transform: uppercase; }
h1 { margin: .25rem 0 .65rem; font-size: clamp(2rem, 4vw, 3.35rem); line-height: 1.08; letter-spacing: -.035em; }
h2 { margin-top: 3rem; font-size: clamp(1.45rem, 2.4vw, 2rem); line-height: 1.2; }
h3 { line-height: 1.25; }
.lede { color: var(--muted); font-size: 1.12rem; }
.run-meta { color: var(--muted); font-size: .9rem; }
code { overflow-wrap: anywhere; }
.concept-map { display: grid; grid-template-columns: minmax(220px, .8fr) minmax(0, 2.6fr); gap: 1rem; align-items: stretch; }
.concept-core, .concept-card, .external-card { border: 1px solid var(--line); border-radius: .8rem; padding: 1.1rem; }
.concept-core { display: flex; flex-direction: column; justify-content: center; background: var(--blue-soft); border-color: #a9c9f7; }
.concept-core strong { font-size: 1.18rem; }
.concept-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.concept-card { background: var(--soft); }
.concept-card.engine { background: var(--purple-soft); }
.concept-card.mode { background: var(--teal-soft); }
.concept-card.method { background: var(--amber-soft); }
.concept-card strong, .external-card strong { display: block; margin-bottom: .25rem; }
.chips { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .65rem; }
.chip { display: inline-block; padding: .18rem .48rem; border: 1px solid #8490a255; border-radius: 999px; background: #ffffffb8; color: var(--ink); font-size: .78rem; }
.external-card { grid-column: 1 / -1; border-style: dashed; border-color: #8b95a3; background: #f1f3f6; }
.section-intro { max-width: 980px; color: var(--muted); }
.metric-guide { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: .7rem 1.25rem; margin: 1.2rem 0; padding: .8rem 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); color: var(--muted); font-size: .88rem; }
.metric-guide strong { display: block; color: var(--ink); }
.data-note { margin: .75rem 0 1.1rem; color: var(--muted); font-size: .82rem; }
.comparison-note { max-width: 1050px; margin: 1rem 0 1.3rem; padding-left: .8rem; border-left: 3px solid #a9c9f7; color: var(--muted); font-size: .92rem; }
.comparison-note strong { color: var(--ink); }
.table-wrap { width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: .65rem; background: var(--surface); }
table { width: 100%; border-collapse: collapse; min-width: 780px; font-size: .9rem; }
th, td { padding: .7rem .75rem; border-bottom: 1px solid var(--line); text-align: right; vertical-align: middle; }
thead th { background: var(--soft); color: #344054; font-size: .8rem; letter-spacing: .015em; text-align: center; }
thead th.row-heading { text-align: left; }
tbody th { min-width: 175px; text-align: left; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover { background: #f8faff; }
.group-heading { text-align: center; }
.external-heading { background: #eceff3; color: var(--external); text-align: center; }
.operation-group th { min-width: 0; padding: .48rem .75rem; border-top: 1px solid #b8c5d8; background: var(--blue-soft); color: #244c83; font-size: .78rem; letter-spacing: .035em; }
.operation-group:hover { background: var(--blue-soft); }
.metric { display: flex; flex-direction: column; align-items: flex-end; white-space: nowrap; }
.metric + .metric { margin-top: .45rem; padding-top: .4rem; border-top: 1px solid var(--line); }
.metric.best { color: #06685f; font-weight: 750; }
.metric-label { color: var(--muted); font-size: .67rem; font-weight: 650; letter-spacing: .04em; text-transform: uppercase; }
.ratio { display: block; color: var(--muted); font-size: .72rem; font-weight: 400; }
.workload-list { display: flex; flex-direction: column; gap: .12rem; text-align: left; }
.workload-heading { text-align: center !important; }
.summary-table { min-width: 1360px; }
.summary-table td:nth-child(n+3) { min-width: 120px; }
.external-cell { background: #fafbfc; }
.unavailable { color: #9aa3b1; }
.workload { color: var(--muted); font-size: .9rem; }
.page-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; margin-top: 1rem; }
.page-card { display: block; min-height: 120px; padding: 1.1rem; border: 1px solid var(--line); border-radius: .75rem; color: var(--ink); text-decoration: none; background: var(--surface); box-shadow: 0 8px 30px #1a2b4b0b; }
.page-card:hover { border-color: #8cb5ef; transform: translateY(-1px); }
.page-card strong { display: block; margin-bottom: .3rem; font-size: 1.05rem; }
.page-card span { color: var(--muted); }
.local-links { display: flex; flex-wrap: wrap; gap: .5rem; margin: 1.2rem 0 2rem; }
.local-links a { padding: .35rem .65rem; border: 1px solid var(--line); border-radius: 999px; text-decoration: none; }
.option-section { scroll-margin-top: 5rem; }
.choice-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 1rem; }
.choice-card { min-width: 0; padding: 1rem; border: 1px solid var(--line); border-radius: .7rem; background: var(--soft); }
.choice-card h3 { margin-top: 0; }
.choice-card table { min-width: 460px; background: var(--surface); }
.choice-table th, .choice-table td { text-align: center; }
.choice-table .metric { align-items: center; }
.scaling-layout { display: grid; grid-template-columns: minmax(230px, 290px) minmax(0, 1fr); gap: 2rem; align-items: start; }
.plot-toc { position: sticky; top: 5.2rem; max-height: calc(100vh - 6.2rem); overflow: auto; padding: 1rem; border: 1px solid var(--line); border-radius: .7rem; background: var(--soft); }
.plot-toc strong { display: block; margin-bottom: .5rem; }
.plot-toc ul { margin: 0; padding-left: 1.2rem; }
.plot-toc li { margin: .42rem 0; }
.scaling-section { scroll-margin-top: 5rem; }
.plot-card { scroll-margin-top: 5rem; margin: 1rem 0; border: 1px solid var(--line); border-radius: .75rem; background: var(--surface); box-shadow: 0 6px 24px #1a2b4b09; }
.plot-card summary { display: flex; justify-content: space-between; gap: 1rem; padding: 1rem 1.1rem; cursor: pointer; list-style-position: outside; }
.plot-card summary strong { display: block; }
.plot-card summary small { display: block; max-width: 780px; margin-top: .25rem; color: var(--muted); }
.plot-content { padding: 0 1rem 1rem; border-top: 1px solid var(--line); }
.plot-content img { display: block; width: 100%; height: auto; margin-top: 1rem; }
.plot-kind { align-self: start; flex: 0 0 auto; }
.history-intro { max-width: 960px; margin-bottom: 1.25rem; }
.history-intro h1 { font-size: clamp(1.8rem, 3vw, 2.6rem); }
.history-frame { display: block; width: 100%; min-height: 720px; height: calc(100vh - 240px); border: 1px solid var(--line); border-radius: .75rem; background: var(--surface); }
.footer-links { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--line); color: var(--muted); }
@media (max-width: 900px) {
  .concept-map, .scaling-layout { grid-template-columns: 1fr; }
  .plot-toc { position: static; max-height: none; }
  .page-cards { grid-template-columns: 1fr; }
}
@media (max-width: 650px) {
  .header-inner { align-items: flex-start; flex-direction: column; }
  .site-nav { justify-content: flex-start; }
  .concept-grid { grid-template-columns: 1fr; }
  main { padding-top: 1.5rem; }
}
"""


def _site_navigation(active_page: str, relative_root: str, include_asv_history: bool) -> str:
    """Return consistent navigation ordered from summary to detailed ASV history."""

    links = [
        ("summary", f"{relative_root}index.html", "Summary"),
        ("options", f"{relative_root}comparisons/index.html", "Options"),
        ("scaling", f"{relative_root}comparisons/{SCALING_REPORT_PAGE}", "Scaling"),
    ]
    if include_asv_history:
        links.append(("history", f"{relative_root}{HISTORY_REPORT_PAGE}", "Performance history"))
    rendered_links = []
    for key, url, label in links:
        current = ' aria-current="page"' if key == active_page else ""
        rendered_links.append(f'<a href="{url}"{current}>{label}</a>')
    return "".join(rendered_links)


def _html_page(
    title: str,
    active_page: str,
    body: str,
    *,
    relative_root: str,
    include_asv_history: bool,
    script: str = "",
) -> str:
    """Wrap one generated view in the shared visual style and navigation."""

    navigation = _site_navigation(active_page, relative_root, include_asv_history)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{html.escape(title)}</title><style>{_SITE_STYLE}</style></head><body>",
            '<header class="site-header"><div class="header-inner">',
            f'<a class="brand" href="{relative_root}index.html">GeoUtils benchmarks</a>',
            f'<nav class="site-nav" aria-label="Benchmark views">{navigation}</nav>',
            "</div></header>",
            f"<main>{body}</main>",
            script,
            "</body></html>",
        ]
    )


def _run_metadata(result: Any) -> str:
    """Return concise commit and machine metadata for a generated page."""

    machine = html.escape(str(result.params.get("machine", "unknown")))
    commit = html.escape(str(result.commit_hash))
    return f'<p class="run-meta">Commit <code>{commit}</code> on machine <code>{machine}</code></p>'


def _metric_guide(*, include_operation: bool = False, include_ratios: bool = True) -> str:
    """Define the metrics used by fixed-workload tables and scaling plots."""

    items = []
    if include_operation:
        items.append(
            "<span><strong>Operation only</strong> Calculation after the execution mode has been initialized.</span>"
        )
    items.extend(
        [
            "<span><strong>Elapsed time</strong> Prepared inputs through completed output, including worker or CLI "
            "startup.</span>",
            "<span><strong>Peak memory</strong> Highest combined memory of the benchmark process and its child "
            "workers.</span>",
        ]
    )
    if include_ratios:
        items.append("<span><strong>Ratios</strong> Relative values for the same workload; lower is better.</span>")
    return f'<div class="metric-guide" aria-label="Metric definitions">{"".join(items)}</div>'


def _gdal_reference_note() -> str:
    """Explain the purpose and limits of the external GDAL comparison."""

    return (
        '<p class="comparison-note"><strong>GDAL CLI reference.</strong> It appears only where a standalone GDAL '
        "command performs an equivalent operation. GeoUtils aims for comparable single-worker costs while supporting "
        "Python workflows, alternative engines and data larger than memory; it need not be faster for every isolated "
        "call. For reprojection, GDAL receives a memory budget based on the GeoUtils chunk size.</p>"
    )


def _data_size_note() -> str:
    """Give compact raw-data context for the representative workload."""

    return (
        '<p class="data-note"><strong>Data-size context:</strong> a one-band 2,048² raster is 16.8 MB as Float32 '
        "(33.6 MB as Float64; 4.2 MB as UInt8), while 17² Float64 XYZ points are about 0.007 MB before object "
        "overhead.</p>"
    )


def _concept_map() -> str:
    """Introduce the four GeoUtils choices and keep the GDAL CLI visibly separate."""

    return "".join(
        [
            '<div class="concept-map">',
            '<div class="concept-core"><strong>GeoUtils operation</strong>',
            "<span>The task being measured, such as reprojection, filtering or gridding.</span></div>",
            '<div class="concept-grid">',
            '<div class="concept-card method"><strong>Method</strong><span>The algorithm selected for a specific '
            'operation.</span><div class="chips"><span class="chip">Gridding — IDW</span>'
            '<span class="chip">Gridding — Circular mean</span>'
            '<span class="chip">Reprojection — Bilinear</span></div></div>',
            '<div class="concept-card engine"><strong>Calculation engine</strong><span>The library GeoUtils uses to '
            'carry out the calculation.</span><div class="chips"><span class="chip">Rasterio / GDAL</span>'
            '<span class="chip">SciPy</span><span class="chip">Numba</span></div></div>',
            '<div class="concept-card mode"><strong>Execution mode</strong><span>How an operation runs: '
            "in-memory or chunked out-of-memory.</span>"
            '<div class="chips"><span class="chip">Eager</span><span class="chip">Dask</span>'
            '<span class="chip">Multiprocessing</span></div></div>',
            '<div class="concept-card"><strong>Chunk strategy</strong><span>How a chunked operation reconciles '
            'separate partial results.</span><div class="chips"><span class="chip">Subsampling — Sequential</span>'
            '<span class="chip">Subsampling — Top-k</span>'
            '<span class="chip">Polygonization — Label stitch</span></div></div>',
            "</div>",
            '<div class="external-card"><strong>External reference: GDAL CLI</strong>',
            "<span>Standalone GDAL file-to-file command run outside GeoUtils. It is distinct from Rasterio/GDAL, "
            "which GeoUtils uses as a calculation engine for some operations.</span></div></div>",
        ]
    )


def _comparison_html(
    result: Any,
    records: list[ComparisonMeasurement],
    *,
    includes_performance_change: bool = False,
    include_asv_history: bool = True,
) -> str:
    """Return representative tables comparing each kind of GeoUtils option."""

    change_link = (
        f'<a href="{PERFORMANCE_CHANGE_REPORT}">View the before/after performance summary</a> · '
        if includes_performance_change
        else ""
    )
    body = "".join(
        [
            '<section class="hero"><span class="eyebrow">Options</span>',
            "<h1>Compare GeoUtils options</h1>",
            '<p class="lede">Each section changes one GeoUtils option at a common workload while keeping the others '
            "fixed. Runs use one worker and one thread.</p>",
            _run_metadata(result),
            '<div class="local-links"><a href="#execution-modes">Execution modes</a>',
            '<a href="#calculation-engines">Calculation engines</a><a href="#operation-methods">Methods</a>',
            '<a href="#chunk-strategies">Chunk strategies</a></div></section>',
            _metric_guide(),
            '<section class="option-section" id="execution-modes">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['execution_mode'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["execution_mode"][1])}</p>',
            _execution_summary_table(records),
            "</section>",
            '<section class="option-section" id="calculation-engines">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['calculation_engine'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["calculation_engine"][1])} GDAL CLI '
            "is shown separately where an equivalent external command exists.</p>",
            _engine_summary_table(records),
            "</section>",
            '<section class="option-section" id="operation-methods">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['method'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["method"][1])} Methods can answer '
            "different numerical questions, so these costs are not a quality ranking.</p>",
            f'<div class="choice-grid">{_choice_tables("method", records)}</div></section>',
            '<section class="option-section" id="chunk-strategies">',
            f"<h2>{html.escape(COMPARISON_SECTION_DETAILS['strategy'][0])}</h2>",
            f'<p class="section-intro">{html.escape(COMPARISON_SECTION_DETAILS["strategy"][1])}</p>',
            f'<div class="choice-grid">{_choice_tables("strategy", records)}</div></section>',
            '<p class="footer-links">',
            change_link,
            '<a href="comparisons.csv">Download CSV</a> · <a href="comparisons.json">Download JSON</a></p>',
        ]
    )
    return _html_page(
        "Compare GeoUtils options",
        "options",
        body,
        relative_root="../",
        include_asv_history=include_asv_history,
    )


def _scaling_html(
    result: Any,
    *,
    include_asv_history: bool = True,
) -> str:
    """Return collapsible plots grouped by the input quantity that grows."""

    dimension_labels = {
        "execution_mode": "Execution modes",
        "calculation_engine": "Calculation engines",
        "method": "Methods",
        "strategy": "Chunk strategies",
    }
    toc = []
    sections = []
    for parameter_label, (section_title, section_description) in SCALING_SECTION_DETAILS.items():
        comparisons = tuple(comparison for comparison in COMPARISONS if comparison.parameter_label == parameter_label)
        if not comparisons:
            continue
        section_id = parameter_label.lower().replace(" ", "-").replace("(", "").replace(")", "")
        plot_links = "".join(
            f'<li><a href="#{html.escape(comparison.slug)}">{html.escape(comparison.title)}</a></li>'
            for comparison in comparisons
        )
        toc.append(f'<li><a href="#{section_id}">{html.escape(section_title)}</a><ul>{plot_links}</ul></li>')
        plots = []
        for comparison in comparisons:
            slug = html.escape(comparison.slug)
            title = html.escape(comparison.title)
            plots.append(
                f'<details class="plot-card" id="{slug}"><summary><span><strong>{title}</strong>'
                f"<small>{html.escape(comparison.description)}</small></span>"
                f'<span class="chip plot-kind">{dimension_labels[comparison.series_dimension]}</span></summary>'
                f'<div class="plot-content"><img loading="lazy" src="{slug}.svg" alt="{title}"></div></details>'
            )
        sections.append(
            f'<section class="scaling-section" id="{section_id}"><h2>{html.escape(section_title)}</h2>'
            f'<p class="section-intro">{html.escape(section_description)}</p>{"".join(plots)}</section>'
        )
    body = "".join(
        [
            '<section class="hero"><span class="eyebrow">Scaling</span>',
            "<h1>Performance as inputs grow</h1>",
            '<p class="lede">Each section varies one input size while other settings stay fixed. Select a comparison '
            "to show or hide its plot.</p>",
            _run_metadata(result),
            "</section>",
            _metric_guide(include_operation=True, include_ratios=False),
            '<div class="scaling-layout"><nav class="plot-toc" aria-label="Scaling sections">',
            "<strong>Input varied</strong><ul>",
            *toc,
            "</ul></nav><div>",
            *sections,
            '<p class="footer-links"><a href="comparisons.csv">Download CSV</a> · '
            '<a href="comparisons.json">Download JSON</a></p></div></div>',
        ]
    )
    script = (
        "<script>function openLinkedPlot(){const id=location.hash.slice(1);const target=document.getElementById(id);"
        "if(target&&target.tagName==='DETAILS')target.open=true;}addEventListener('hashchange',openLinkedPlot);"
        "openLinkedPlot();</script>"
    )
    return _html_page(
        "GeoUtils scaling benchmarks",
        "scaling",
        body,
        relative_root="../",
        include_asv_history=include_asv_history,
        script=script,
    )


def _history_html(result: Any) -> str:
    """Embed the native ASV report below the shared GeoUtils navigation."""

    body = "".join(
        [
            '<section class="history-intro"><span class="eyebrow">History</span>',
            "<h1>Performance history</h1>",
            '<p class="lede">Explore ASV measurements across commits, including its benchmark grid, detailed '
            "history plots and detected regressions.</p>",
            _run_metadata(result),
            "</section>",
            '<iframe class="history-frame" src="asv/index.html" title="ASV performance history"></iframe>',
            '<p class="footer-links"><a href="asv/index.html">Open the native ASV report</a></p>',
        ]
    )
    return _html_page(
        "GeoUtils performance history",
        "history",
        body,
        relative_root="",
        include_asv_history=True,
    )


def _site_index(
    result: Any,
    records: list[ComparisonMeasurement],
    *,
    includes_performance_change: bool = False,
    include_asv_history: bool = True,
) -> str:
    """Return the benchmark landing page with definitions and one headline table."""

    cards = [
        '<a class="page-card" href="comparisons/index.html"><strong>Compare options</strong>'
        "<span>Isolate execution modes, calculation engines, methods and chunk strategies at one workload.</span></a>",
        f'<a class="page-card" href="comparisons/{SCALING_REPORT_PAGE}"><strong>Scaling</strong>'
        "<span>See how elapsed time and peak memory change as raster, chunk, point or sample size grows.</span></a>",
    ]
    if include_asv_history:
        cards.append(
            f'<a class="page-card" href="{HISTORY_REPORT_PAGE}"><strong>Performance history</strong>'
            "<span>Inspect detailed ASV measurements and changes across commits.</span></a>"
        )
    change_link = (
        f'<p><a href="comparisons/{PERFORMANCE_CHANGE_REPORT}">View the before/after branch summary</a></p>'
        if includes_performance_change
        else ""
    )
    body = "".join(
        [
            '<section class="hero"><span class="eyebrow">Summary</span>',
            "<h1>GeoUtils performance benchmarks</h1>",
            '<p class="lede">Representative time and memory measurements for raster, vector and point cloud '
            "operations.</p>",
            _run_metadata(result),
            "</section>",
            "<section><h2>How GeoUtils operations vary</h2>",
            '<p class="section-intro">Each comparison changes one GeoUtils option at a time. GDAL CLI is a separate '
            "external reference.</p>",
            _concept_map(),
            "</section>",
            "<section><h2>Performance table</h2>",
            '<p class="section-intro">Each row shows elapsed time and peak memory for a fixed workload (operation '
            "parameters and input size). All options run without parallelism: one computational process using one "
            "thread.</p>",
            _gdal_reference_note(),
            _metric_guide(include_ratios=False),
            _data_size_note(),
            _headline_summary_table(records),
            "</section>",
            change_link,
            '<section><h2>Explore the results further</h2><div class="page-cards">',
            *cards,
            "</div></section>",
            '<p class="footer-links"><a href="comparisons/comparisons.csv">Download CSV</a> · '
            '<a href="comparisons/comparisons.json">Download JSON</a></p>',
        ]
    )
    return _html_page(
        "GeoUtils benchmark results",
        "summary",
        body,
        relative_root="",
        include_asv_history=include_asv_history,
    )


def render_comparisons(
    result: Any,
    website_directory: Path,
    *,
    baseline_result: Any | None = None,
    include_asv_history: bool = True,
) -> list[ComparisonMeasurement]:
    """Write comparison plots and one landing page beside the native ASV site."""

    # Only expose the history shell when ASV has already published its native report
    include_asv_history = include_asv_history and (website_directory / "asv" / "index.html").is_file()

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
        _comparison_html(
            result,
            records,
            includes_performance_change=includes_performance_change,
            include_asv_history=include_asv_history,
        ),
        encoding="utf-8",
    )
    (report_directory / SCALING_REPORT_PAGE).write_text(
        _scaling_html(result, include_asv_history=include_asv_history),
        encoding="utf-8",
    )

    # The custom summary stays separate from files generated internally by ASV
    website_directory.mkdir(parents=True, exist_ok=True)
    (website_directory / "index.html").write_text(
        _site_index(
            result,
            records,
            includes_performance_change=includes_performance_change,
            include_asv_history=include_asv_history,
        ),
        encoding="utf-8",
    )
    if include_asv_history:
        (website_directory / HISTORY_REPORT_PAGE).write_text(_history_html(result), encoding="utf-8")
    return records


def _render_preview(website_directory: Path) -> None:
    """Write a complete comparison website using deterministic fake measurements."""

    # Two fake revisions also exercise the generated before/after performance table
    current_result = _PreviewResult()
    baseline_result = _PreviewResult(commit_hash="preview-baseline", geoutils_scale=1.25)
    render_comparisons(
        current_result,
        website_directory,
        baseline_result=baseline_result,
        include_asv_history=(website_directory / "asv" / "index.html").is_file(),
    )


def main() -> None:
    """Select one saved ASV result and render its requested report outputs."""

    # This command runs explicitly after asv publish and is not called during ASV measurement
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=Path("benchmarks/results/asv/results"))
    parser.add_argument("--website-directory", type=Path)
    parser.add_argument("--doc-dir", type=Path)
    parser.add_argument("--doc-only", action="store_true")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="render a complete website from deterministic fake data",
    )
    parser.add_argument("--machine")
    parser.add_argument("--commit")
    parser.add_argument("--baseline-commit")
    parser.add_argument("--environment")
    args = parser.parse_args()

    # Preview mode needs neither saved ASV results nor benchmark execution
    if args.preview:
        website_directory = args.website_directory or PREVIEW_WEBSITE_DIRECTORY
        _render_preview(website_directory)
        print(f"Benchmark preview written to {website_directory / 'index.html'}")
        return

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
        website_directory = args.website_directory or Path("benchmarks/results/asv/html")
        render_comparisons(result, website_directory, baseline_result=baseline_result)
    if args.doc_dir is not None:
        render_documentation_snapshot(result, args.doc_dir)
    elif args.doc_only:
        parser.error("--doc-only requires --doc-dir")


if __name__ == "__main__":
    main()
