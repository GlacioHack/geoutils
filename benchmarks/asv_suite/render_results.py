"""Render saved ASV measurements for the benchmark website and user documentation."""

from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.asv_suite.comparisons import (
    COMPARISONS,
    Comparison,
)

COMPARISON_BENCHMARK_MODULE = "asv_suite.comparisons"
COMPARISON_REPORT_DIRECTORY = "comparisons"
DOCUMENTATION_TIME_PLOT = "time_relative_to_gdal.svg"
DOCUMENTATION_MEMORY_PLOT = "peak_ram_by_raster_size.svg"
DOCUMENTATION_DATA = "benchmark_snapshot.json"
DOCUMENTATION_SUMMARY = "summary.md.inc"

# Colors stay consistent between the detailed ASV plots and the concise documentation snapshot
IMPLEMENTATION_COLORS = {
    "Eager": "#0072B2",
    "Dask": "#E69F00",
    "Multiprocessing": "#009E73",
    "GDAL": "#6C6C6C",
}


@dataclass(frozen=True)
class ImplementationMeasurement:
    """Store one implementation result at one numeric parameter value."""

    comparison: str
    implementation: str
    parameter: int
    operation_time_s: float
    end_to_end_time_s: float
    peak_process_tree_rss_mb: float


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


def _required_benchmark_keys() -> set[str]:
    """Return every saved result needed to render all comparisons."""

    # A partially interrupted ASV run must not replace the latest complete documentation result
    methods = ("time_operation", "track_end_to_end_time_s", "track_peak_process_tree_rss_mb")
    return {
        _benchmark_key(class_name, method)
        for comparison in COMPARISONS
        for _, class_name in comparison.series
        for method in methods
    }


def select_asv_result(
    results_directory: Path,
    *,
    machine: str | None = None,
    commit: str | None = None,
    environment: str | None = None,
) -> Any:
    """Select the newest complete comparison result matching optional filters."""

    # Import ASV only when rendering so benchmark discovery remains lightweight
    from asv.results import iter_results

    required_keys = _required_benchmark_keys()
    candidates = []
    for result in iter_results(str(results_directory)):
        # Machine and commit filters make local multi-machine histories predictable
        if machine is not None and result.params.get("machine") != machine:
            continue
        if commit is not None and not result.commit_hash.startswith(commit):
            continue

        # ASV does not expose the environment name as a public property
        result_environment = str(getattr(result, "_env_name", ""))
        if environment is not None and result_environment != environment:
            continue
        if not required_keys.issubset(result.get_all_result_keys()):
            continue
        candidates.append(result)

    if not candidates:
        raise RuntimeError("No saved ASV result contains all comparisons")
    return max(candidates, key=_latest_timestamp)


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


def collect_implementation_measurements(result: Any) -> list[ImplementationMeasurement]:
    """Combine matching operation-time, end-to-end-time and peak-RAM results."""

    records = []
    available = set(result.get_all_result_keys())
    for comparison in COMPARISONS:
        for implementation, class_name in comparison.series:
            # The three ASV methods share the same numeric parameter values
            keys = {
                "operation": _benchmark_key(class_name, "time_operation"),
                "end_to_end": _benchmark_key(class_name, "track_end_to_end_time_s"),
                "memory": _benchmark_key(class_name, "track_peak_process_tree_rss_mb"),
            }
            missing = set(keys.values()) - available
            if missing:
                raise ValueError(f"Missing comparison results: {', '.join(sorted(missing))}")

            operation_params, operation_times = _result_series(result, keys["operation"])
            end_to_end_params, end_to_end_times = _result_series(result, keys["end_to_end"])
            memory_params, peak_memory = _result_series(result, keys["memory"])
            if operation_params != end_to_end_params or operation_params != memory_params:
                raise ValueError(f"Time and RAM parameters differ for {class_name}")

            # One record per point keeps JSON and CSV useful without plotting software
            records.extend(
                ImplementationMeasurement(
                    comparison.slug,
                    implementation,
                    parameter,
                    operation_time,
                    end_to_end_time,
                    memory,
                )
                for parameter, operation_time, end_to_end_time, memory in zip(
                    operation_params,
                    operation_times,
                    end_to_end_times,
                    peak_memory,
                )
            )
    return records


def _plot_comparison(
    comparison: Comparison,
    records: list[ImplementationMeasurement],
    output: Path,
) -> None:
    """Write aligned operation-time, end-to-end-time and peak-RAM plots."""

    # Keep Matplotlib caches in a writable disposable location under sandboxed runs
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geoutils-matplotlib-cache"))

    # Import plotting only for explicit rendering commands
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout="constrained")
    for implementation, _ in comparison.series:
        # Stable parameter sorting keeps lines readable if ASV changes storage order
        selected = sorted(
            (
                record
                for record in records
                if record.comparison == comparison.slug and record.implementation == implementation
            ),
            key=lambda record: record.parameter,
        )
        parameters = [record.parameter for record in selected]
        axes[0].plot(parameters, [record.operation_time_s for record in selected], marker="o", label=implementation)
        axes[1].plot(parameters, [record.end_to_end_time_s for record in selected], marker="o", label=implementation)
        axes[2].plot(
            parameters,
            [record.peak_process_tree_rss_mb for record in selected],
            marker="o",
            label=implementation,
        )

    # Separate panels keep initialized work, complete execution and memory unambiguous
    axes[0].set_title(comparison.title)
    axes[0].set_ylabel("Operation time (s)")
    axes[1].set_ylabel("End-to-end time (s)")
    axes[2].set_ylabel("Peak process-tree RSS (MB)")
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
    """Return raster-size comparisons that provide a GDAL reference line."""

    return tuple(
        comparison
        for comparison in COMPARISONS
        if comparison.documentation and any(implementation == "GDAL" for implementation, _ in comparison.series)
    )


def _largest_shared_parameter(
    comparison: Comparison,
    records: list[ImplementationMeasurement],
) -> int:
    """Return the largest parameter measured by every implementation in one comparison."""

    # Shared parameters keep the normalized time bars based on exactly the same input size
    parameters_by_implementation = []
    for implementation, _ in comparison.series:
        parameters_by_implementation.append(
            {
                record.parameter
                for record in records
                if record.comparison == comparison.slug and record.implementation == implementation
            }
        )
    shared_parameters = set.intersection(*parameters_by_implementation)
    if not shared_parameters:
        raise ValueError(f"No shared parameter found for {comparison.slug}")
    return max(shared_parameters)


def _measurement_at(
    records: list[ImplementationMeasurement],
    comparison: str,
    implementation: str,
    parameter: int,
) -> ImplementationMeasurement:
    """Return one uniquely identified implementation measurement."""

    selected = [
        record
        for record in records
        if record.comparison == comparison and record.implementation == implementation and record.parameter == parameter
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {implementation} result for {comparison} at parameter {parameter}, found {len(selected)}"
        )
    return selected[0]


def _documentation_operation_name(comparison: Comparison) -> str:
    """Return the concise operation name used on documentation graphics."""

    return comparison.title.removesuffix(" raster size")


def _plot_time_relative_to_gdal(
    records: list[ImplementationMeasurement],
    output: Path,
) -> None:
    """Compare end-to-end backend time with GDAL on the largest shared raster."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    implementations = ("Eager", "Dask", "Multiprocessing")
    bar_width = 0.24
    positions = list(range(len(comparisons)))
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")

    for implementation_index, implementation in enumerate(implementations):
        ratios = []
        for comparison in comparisons:
            parameter = _largest_shared_parameter(comparison, records)
            gdal_time = _measurement_at(records, comparison.slug, "GDAL", parameter).end_to_end_time_s
            backend_time = _measurement_at(
                records,
                comparison.slug,
                implementation,
                parameter,
            ).end_to_end_time_s
            ratios.append(backend_time / gdal_time)

        # Group implementations around each operation so their GDAL ratio is easy to compare
        offset = (implementation_index - 1) * bar_width
        bars = axis.bar(
            [position + offset for position in positions],
            ratios,
            width=bar_width,
            label=implementation,
            color=IMPLEMENTATION_COLORS[implementation],
        )
        axis.bar_label(bars, fmt="%.1f×", padding=3, fontsize=8)

    # GDAL equals one by definition and remains visible when every backend is slower
    axis.axhline(1.0, color=IMPLEMENTATION_COLORS["GDAL"], linestyle="--", linewidth=1.2, label="GDAL")
    axis.set_xticks(positions, [_documentation_operation_name(comparison) for comparison in comparisons])
    axis.set_ylabel("End-to-end time relative to GDAL")
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=4)
    figure.savefig(output, format="svg")
    plt.close(figure)


def _plot_peak_ram_by_raster_size(
    records: list[ImplementationMeasurement],
    output: Path,
) -> None:
    """Compare total process-tree RAM as the raster size increases."""

    # Import plotting only for an explicit rendering command
    import matplotlib.pyplot as plt

    comparisons = _gdal_comparisons()
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for axis, comparison in zip(axes.flat, comparisons):
        for implementation, _ in comparison.series:
            # Sorting protects the displayed dependency from ASV storage order
            selected = sorted(
                (
                    record
                    for record in records
                    if record.comparison == comparison.slug and record.implementation == implementation
                ),
                key=lambda record: record.parameter,
            )
            axis.plot(
                [record.parameter for record in selected],
                [record.peak_process_tree_rss_mb for record in selected],
                marker="o",
                label=implementation,
                color=IMPLEMENTATION_COLORS[implementation],
            )

        axis.set_title(_documentation_operation_name(comparison))
        axis.set_xlabel("Raster width and height (pixels)")
        axis.set_ylabel("Peak process-tree RSS (MB)")
        axis.grid(alpha=0.3)

    # One legend applies to every panel and leaves the operation curves uncluttered
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=4)
    figure.savefig(output, format="svg", bbox_inches="tight")
    plt.close(figure)


def _result_payload(result: Any, records: list[ImplementationMeasurement]) -> dict[str, Any]:
    """Return measurements and enough run metadata to interpret them later."""

    return {
        "metadata": {
            "commit": result.commit_hash,
            "date": result.date,
            "environment": str(getattr(result, "_env_name", "")),
            "machine": result.params,
            "operation_time": "wall-clock time after implementation initialization",
            "end_to_end_time": "wall-clock time including implementation initialization but excluding input generation",
            "memory": "peak aggregate process-tree RSS",
        },
        "measurements": [asdict(record) for record in records],
    }


def _documentation_summary(result: Any) -> str:
    """Return the short MyST fragment embedded in the performance page."""

    commit = str(result.commit_hash)
    machine = str(result.params.get("machine", "unknown"))
    return "\n".join(
        [
            ":::{figure} /imgs/benchmarking/time_relative_to_gdal.svg",
            ":alt: End-to-end GeoUtils backend time relative to GDAL for four raster operations",
            "",
            "End-to-end time on the largest raster size shared by every implementation. GDAL is the reference at one.",
            ":::",
            "",
            ":::{figure} /imgs/benchmarking/peak_ram_by_raster_size.svg",
            ":alt: Peak process-tree RAM by raster size for GeoUtils backends and GDAL",
            "",
            "Peak RAM for the benchmark process and all implementation workers as raster dimensions increase.",
            ":::",
            "",
            f"Measured at GeoUtils commit [`{commit[:8]}`]"
            f"(https://github.com/GlacioHack/geoutils/commit/{commit}) on `{machine}`. "
            "{download}`Download the exact values and run metadata </imgs/benchmarking/benchmark_snapshot.json>`.",
            "",
        ]
    )


def render_documentation_snapshot(
    result: Any,
    output_directory: Path,
) -> list[ImplementationMeasurement]:
    """Write the two reviewed reference graphics and their complete numeric source."""

    # Keep Matplotlib caches in a writable disposable location under sandboxed runs
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "geoutils-matplotlib-cache"))
    output_directory.mkdir(parents=True, exist_ok=True)
    records = collect_implementation_measurements(result)

    # Documentation uses two broad summaries while the ASV site retains every detailed panel
    _plot_time_relative_to_gdal(records, output_directory / DOCUMENTATION_TIME_PLOT)
    _plot_peak_ram_by_raster_size(records, output_directory / DOCUMENTATION_MEMORY_PLOT)
    (output_directory / DOCUMENTATION_DATA).write_text(
        json.dumps(_result_payload(result, records), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_directory / DOCUMENTATION_SUMMARY).write_text(_documentation_summary(result), encoding="utf-8")
    return records


def _markdown_report(result: Any) -> str:
    """Return a compact Markdown index linking plots and numeric results."""

    machine = result.params.get("machine", "unknown")
    lines = [
        "# GeoUtils comparisons",
        "",
        f"Commit `{result.commit_hash}` on machine `{machine}`",
        "",
        "Operation time excludes implementation initialization. End-to-end time includes it but excludes input "
        "generation. Peak RAM is aggregate RSS for the benchmark process and all implementation children.",
        "",
        "Raw values: [CSV](comparisons.csv) · [JSON](comparisons.json)",
    ]
    for comparison in COMPARISONS:
        lines.extend(["", f"## {comparison.title}", "", f"![{comparison.title}]({comparison.slug}.svg)"])
    return "\n".join(lines) + "\n"


def _comparison_html(result: Any) -> str:
    """Return the standalone comparison browser page."""

    machine = html.escape(str(result.params.get("machine", "unknown")))
    commit = html.escape(str(result.commit_hash))
    sections = []
    for comparison in COMPARISONS:
        title = html.escape(comparison.title)
        sections.append(f'<section><h2>{title}</h2><img src="{comparison.slug}.svg" alt="{title}"></section>')
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
            "<p>Operation time excludes implementation initialization. End-to-end time includes it but excludes "
            "input generation. Peak RAM is aggregate RSS for the benchmark process and all implementation children.</p>",
            '<p>Raw values: <a href="comparisons.csv">CSV</a> · ' '<a href="comparisons.json">JSON</a></p>',
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
) -> list[ImplementationMeasurement]:
    """Write comparison plots and one landing page beside the native ASV site."""

    # All generated material stays under the gitignored ASV website directory
    report_directory = website_directory / COMPARISON_REPORT_DIRECTORY
    report_directory.mkdir(parents=True, exist_ok=True)
    records = collect_implementation_measurements(result)
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
    (report_directory / "README.md").write_text(_markdown_report(result), encoding="utf-8")
    (report_directory / "index.html").write_text(_comparison_html(result), encoding="utf-8")

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
    parser.add_argument("--environment")
    args = parser.parse_args()

    # Optional filters select a reproducible record from histories with several machines
    result = select_asv_result(
        args.results_directory,
        machine=args.machine,
        commit=args.commit,
        environment=args.environment,
    )
    if not args.doc_only:
        render_comparisons(result, args.website_directory)
    if args.doc_dir is not None:
        render_documentation_snapshot(result, args.doc_dir)
    elif args.doc_only:
        parser.error("--doc-only requires --doc-dir")


if __name__ == "__main__":
    main()
