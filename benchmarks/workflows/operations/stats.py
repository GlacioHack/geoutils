"""Define raster and grouped-statistics benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from benchmarks.flox_comparison.runner import _GroupedFloxBenchmark
from benchmarks.workflows.config import (
    GROUP_COUNT_AXIS,
    RASTER_AXIS,
    RASTER_CHUNK_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
    external_case,
    merge_cases,
    raster_size_config,
    strategy_cases,
)
from geoutils._misc import import_optional

ORDER = 10
STRATEGIES = ("auto", "dense", "sparse", "groupwise")

############################
# Operation execution
############################


def statistics_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public options for regular or grouped statistics."""

    if case.operation == "statistics":
        return {"statistics": ["mean", "std", "valid count"]}
    statistics = ["mean", "std", "min", "max"] if case.method == "moments" else ["median", "nmad"]
    return {"statistics": statistics, "strategy": case.strategy or "auto"}


def run_statistics(runner: Any, case: BenchmarkCase) -> float:
    """Compute regular reductions or grouped statistics on the prepared raster."""

    raster = runner.make_raster()
    if case.operation == "statistics":
        if runner.backend != "dask":
            raise ValueError("Raster statistics are only registered for Dask")

        # Compute selected reductions without evaluating unrelated quantiles
        import_optional("dask", extra_name="benchmark")
        import dask

        statistics = raster.rst.stats(statistics_options(case, runner.config)["statistics"])
        mean, _, _ = dask.compute(*statistics.values())
        return float(mean)

    return grouped_statistics(runner, raster, case)


def grouped_statistics(runner: Any, raster: Any, case: BenchmarkCase) -> float:
    """Compute grouped moments or exact robust estimates with independent gaps.

    Local groups occupy rectangular regions; interleaved groups span the entire input. Dask builds all value
    and membership arrays lazily. Multiprocessing benchmarks the current array interface, which loads values
    in the client before tiling them for workers. The returned fingerprint checks complete finite counts.
    """

    # Generate coordinates with the same execution backend as the input raster
    height, width = runner.config.shape
    if runner.backend == "dask":
        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        rows = da.arange(height, chunks=runner.config.chunks[0])[:, None]
        columns = da.arange(width, chunks=runner.config.chunks[1])[None, :]
    else:
        rows = np.arange(height)[:, None]
        columns = np.arange(width)[None, :]
    regions = runner.config.grouped_regions_per_axis
    if regions < 1 or regions > min(height, width):
        raise ValueError("Grouped regions per axis must fit within the raster dimensions.")

    # Separate localized membership from groups repeated through every chunk
    if runner.config.grouped_layout == "local":
        groups = (rows * regions // height) * regions + columns * regions // width
    else:
        groups = (rows % regions) * regions + columns % regions
    positions = rows * width + columns
    base = raster.data.squeeze()
    signal = base + (rows % 97) * 0.125 + (columns % 53) * 0.25
    values = {
        "signal": np.where(positions % 17 != 0, signal, np.nan),
        "offset": np.where(positions % 29 != 0, 2 * signal + 10, np.nan),
    }

    # Measure the complete public calculation, including exact group gathering when requested
    from geoutils.stats import stats

    options = statistics_options(case, runner.config)
    mp_config = runner._multiproc_config(case.operation) if runner.backend == "multiprocessing" else None
    result = stats(
        values,
        by={"zone": groups},
        categories={"zone": range(regions**2)},
        **options,
        mp_config=mp_config,
    )

    # Every pixel belongs to a group; missing values follow independent, analytically known periods
    count = height * width
    for name, period in (("signal", 17), ("offset", 29)):
        expected = count - (count + period - 1) // period
        if result[(name, "count")].sum() != expected:
            raise AssertionError(f"Grouped benchmark lost finite observations in {name!r}.")
        if not np.isfinite(result[name].to_numpy()).all():
            raise AssertionError(f"Grouped benchmark returned invalid estimates in {name!r}.")
    return 1.0


# Multiprocessing currently tiles arrays already resident in the client, so only Dask is out of core
OPERATIONS = (
    Operation(
        "statistics",
        run_statistics,
        statistics_options,
        call_name=".stats",
        coverage=OperationCoverage(6, ("dask",)),
    ),
    Operation(
        "grouped_stats",
        run_statistics,
        statistics_options,
        ("statistics", "strategy"),
        {"moments": ("numpy",), "robust": ("numpy",)},
        "moments",
        STRATEGIES,
        "auto",
        call_name="stats",
        coverage=OperationCoverage(7, ("dask",)),
    ),
)


############################
# Cases and input sweeps
############################


# Each case fixes one method, strategy and execution mode for one ASV result series; its sweep owns the changing input
# The sweeps isolate input size, chunk size, membership layout and group count for shared grouped-statistic kernels
EXECUTION_CASES = execution_cases(
    "grouped_stats",
    "moments",
    "numpy",
    strategy="dense",
    pr_executions=("inmem", "dask", "multiprocessing"),
)
# Check each distinct layout and the automatic sparse threshold with a bounded pull-request workload
PR_STRATEGIES = {
    "grouped-stats-chunk-size": ("dense",),
    "grouped-stats-interleaved-chunks": ("groupwise",),
    "grouped-stats-group-count": ("sparse", "auto"),
}
STRATEGY_CASES = {
    sweep_id: strategy_cases(
        "grouped_stats",
        "moments",
        "numpy",
        STRATEGIES,
        execution="dask",
        pr_strategies=PR_STRATEGIES.get(sweep_id, ()),
        variant="interleaved" if sweep_id == "grouped-stats-interleaved-chunks" else None,
    )
    for sweep_id in (
        "grouped-stats-raster-size",
        "grouped-stats-chunk-size",
        "grouped-stats-interleaved-chunks",
        "grouped-stats-group-count",
    )
}
ROBUST_CASES = execution_cases(
    "grouped_stats",
    "robust",
    "numpy",
    strategy="groupwise",
    pr_executions=("dask", "multiprocessing"),
)

# Compare public GeoUtils stats() with direct Flox reductions on the same prepared arrays
FLOX_CASES = {
    sweep_id: execution_cases(
        "grouped_stats",
        "moments",
        "numpy",
        strategy="auto",
        pr_executions=("inmem", "dask", "multiprocessing"),
        variant="flox",
    )
    for sweep_id in ("grouped-flox-raster-size", "grouped-flox-group-count")
}
FLOX_REFERENCES = {
    sweep_id: tuple(
        external_case(
            FLOX_CASES[sweep_id],
            reference="flox",
            pr_check=True,
            execution=execution,
        )
        for execution in ("inmem", "dask")
    )
    for sweep_id in FLOX_CASES
}


def chunk_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Include uneven edge chunks and groups crossing partition boundaries."""

    size = 1_000 if pr_check else 2_000
    chunk_size = int(parameter)
    return {"shape": (size, size), "chunks": (chunk_size, chunk_size)}


def interleaved_chunk_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Keep observations interleaved across all chunks for each tested partition size."""

    return {**chunk_size(parameter, case, pr_check), "grouped_layout": "interleaved"}


def group_count(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Include 4225 groups so automatic reduction exercises its sparse branch."""

    size = 1_000 if pr_check else 2_000
    return {"shape": (size, size), "chunks": (500, 500), "grouped_regions_per_axis": int(parameter)}


def flox_raster_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Vary raster size around 256 local groups and fixed spatial chunks."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000), "grouped_regions_per_axis": 20}


def flox_group_count(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Vary declared groups on a fixed raster with membership repeated across chunks."""

    size = 1_000 if pr_check else 2_000
    return {
        "shape": (size, size),
        "chunks": (1_000, 1_000),
        "grouped_regions_per_axis": int(parameter),
        "grouped_layout": "interleaved",
    }


# Measure the selected cases over each numeric axis, including matched Flox cases for the first two sweeps
SWEEPS = (
    Sweep(
        RASTER_AXIS,
        flox_raster_size,
        FLOX_CASES["grouped-flox-raster-size"],
        FLOX_REFERENCES["grouped-flox-raster-size"],
        harness=_GroupedFloxBenchmark,
        name="grouped-flox",
    ),
    Sweep(
        GROUP_COUNT_AXIS,
        flox_group_count,
        FLOX_CASES["grouped-flox-group-count"],
        FLOX_REFERENCES["grouped-flox-group-count"],
        harness=_GroupedFloxBenchmark,
        name="grouped-flox",
    ),
    Sweep(
        RASTER_AXIS,
        raster_size_config,
        merge_cases(EXECUTION_CASES, STRATEGY_CASES["grouped-stats-raster-size"]),
    ),
    Sweep(
        RASTER_CHUNK_AXIS,
        chunk_size,
        STRATEGY_CASES["grouped-stats-chunk-size"],
    ),
    Sweep(
        RASTER_CHUNK_AXIS,
        interleaved_chunk_size,
        STRATEGY_CASES["grouped-stats-interleaved-chunks"],
        name="grouped-stats-interleaved",
    ),
    Sweep(
        GROUP_COUNT_AXIS,
        group_count,
        STRATEGY_CASES["grouped-stats-group-count"],
    ),
    Sweep(RASTER_AXIS, raster_size_config, ROBUST_CASES, name="grouped-stats-robust"),
)


############################
# Report comparisons
############################

# Comparisons select saved GeoUtils and Flox series for external plots, then GeoUtils series for internal strategy plots
COMPARISONS = (
    *tuple(comparison(sweep, documentation=False, summary=False) for sweep in SWEEPS[:2]),
    comparison(
        SWEEPS[2],
        cases=EXECUTION_CASES,
        slug="grouped-stats-execution-size",
        documentation=False,
    ),
    *tuple(
        comparison(
            sweep,
            cases=STRATEGY_CASES[sweep_id],
            documentation=False,
        )
        for sweep, sweep_id in zip(
            SWEEPS[2:6],
            (
                "grouped-stats-raster-size",
                "grouped-stats-chunk-size",
                "grouped-stats-interleaved-chunks",
                "grouped-stats-group-count",
            ),
            strict=True,
        )
    ),
    comparison(SWEEPS[6], documentation=False),
)
