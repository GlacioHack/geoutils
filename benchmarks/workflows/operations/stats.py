"""Define raster and grouped-statistics benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from benchmarks.flox_comparison.runner import _GroupedFloxBenchmark
from benchmarks.workflows.config import (
    GROUP_COUNT_AXIS,
    RASTER_AXIS,
    RASTER_CHUNK_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    ExternalReferenceCase,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
    external_case,
    merge_cases,
    strategy_cases,
)
from geoutils._misc import import_optional

ORDER = 10
_STRATEGIES = ("auto", "dense", "sparse", "groupwise")

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

    return _grouped_statistics(runner, raster, case)


def _grouped_statistics(runner: Any, raster: Any, case: BenchmarkCase) -> float:
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


OPERATIONS = (
    Operation("statistics", run_statistics, statistics_options, call_name=".stats"),
    Operation(
        "grouped_stats",
        run_statistics,
        statistics_options,
        ("statistics", "strategy"),
        {"moments": ("numpy",), "robust": ("numpy",)},
        "moments",
        _STRATEGIES,
        "auto",
        call_name="stats",
    ),
)

# Multiprocessing currently tiles arrays already resident in the client, so only Dask is out of core
COVERAGE = (
    OperationCoverage("statistics", ("dask",), 1, 6),
    OperationCoverage("grouped_stats", ("dask",), 1, 7),
)


############################
# Cases and input sweeps
############################


# Isolate input size, chunk size, membership layout and group count for shared grouped-statistic kernels
_EXECUTION_CASES = execution_cases(
    "grouped-stats-raster-size",
    "grouped_stats",
    "moments",
    "numpy",
    strategy="dense",
    pr_executions=("eager", "dask", "multiprocessing"),
)
_STRATEGY_CASES = {
    sweep_id: strategy_cases(
        sweep_id,
        "grouped_stats",
        "moments",
        "numpy",
        _STRATEGIES,
        execution="dask",
    )
    for sweep_id in (
        "grouped-stats-raster-size",
        "grouped-stats-chunk-size",
        "grouped-stats-interleaved-chunks",
        "grouped-stats-group-count",
    )
}
_ROBUST_CASES = execution_cases(
    "grouped-stats-robust-size",
    "grouped_stats",
    "robust",
    "numpy",
    strategy="groupwise",
    pr_executions=("dask", "multiprocessing"),
)

# Check each distinct layout and the automatic sparse threshold with a bounded pull-request workload
for _sweep_id, _cases in _STRATEGY_CASES.items():
    _STRATEGY_CASES[_sweep_id] = tuple(
        replace(case, pr_check=True)
        if (_sweep_id, case.strategy)
        in {
            ("grouped-stats-chunk-size", "dense"),
            ("grouped-stats-interleaved-chunks", "groupwise"),
            ("grouped-stats-group-count", "sparse"),
            ("grouped-stats-group-count", "auto"),
        }
        else case
        for case in _cases
    )

# Compare public GeoUtils stats() with direct Flox reductions on the same prepared arrays
_FLOX_CASES = {
    sweep_id: execution_cases(
        sweep_id,
        "grouped_stats",
        "moments",
        "numpy",
        strategy="auto",
        pr_executions=("eager", "dask", "multiprocessing"),
    )
    for sweep_id in ("grouped-flox-raster-size", "grouped-flox-group-count")
}
_FLOX_REFERENCES = {
    sweep_id: tuple(
        external_case(
            _FLOX_CASES[sweep_id],
            reference="flox",
            pr_check=True,
            execution=execution,
        )
        for execution in ("eager", "dask")
    )
    for sweep_id in _FLOX_CASES
}


def _raster_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Prepare two values and 64 spatial groups on the selected raster size."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000)}


def _chunk_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Include uneven edge chunks and groups crossing partition boundaries."""

    size = 1_000 if pr_check else 2_000
    chunk_size = int(parameter)
    return {"shape": (size, size), "chunks": (chunk_size, chunk_size)}


def _interleaved_chunk_size(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Keep observations interleaved across all chunks for each tested partition size."""

    return {**_chunk_size(parameter, case, pr_check), "grouped_layout": "interleaved"}


def _group_count(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Include 4225 groups so automatic reduction exercises its sparse branch."""

    size = 1_000 if pr_check else 2_000
    return {"shape": (size, size), "chunks": (500, 500), "grouped_regions_per_axis": int(parameter)}


def _flox_raster_size(
    parameter: Parameter, case: BenchmarkCase | ExternalReferenceCase, pr_check: bool
) -> Mapping[str, Any]:
    """Vary raster size around 256 local groups and fixed spatial chunks."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000), "grouped_regions_per_axis": 20}


def _flox_group_count(
    parameter: Parameter, case: BenchmarkCase | ExternalReferenceCase, pr_check: bool
) -> Mapping[str, Any]:
    """Vary declared groups on a fixed raster with membership repeated across chunks."""

    size = 1_000 if pr_check else 2_000
    return {
        "shape": (size, size),
        "chunks": (1_000, 1_000),
        "grouped_regions_per_axis": int(parameter),
        "grouped_layout": "interleaved",
    }


SWEEPS = (
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _flox_raster_size,
        _FLOX_CASES["grouped-flox-raster-size"],
        _FLOX_REFERENCES["grouped-flox-raster-size"],
        harness=_GroupedFloxBenchmark,
    ),
    Sweep(
        "groups_per_axis",
        GROUP_COUNT_AXIS,
        _flox_group_count,
        _FLOX_CASES["grouped-flox-group-count"],
        _FLOX_REFERENCES["grouped-flox-group-count"],
        harness=_GroupedFloxBenchmark,
    ),
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _raster_size,
        merge_cases(_EXECUTION_CASES, _STRATEGY_CASES["grouped-stats-raster-size"]),
    ),
    Sweep(
        "chunk_size",
        RASTER_CHUNK_AXIS,
        _chunk_size,
        _STRATEGY_CASES["grouped-stats-chunk-size"],
    ),
    Sweep(
        "chunk_size",
        RASTER_CHUNK_AXIS,
        _interleaved_chunk_size,
        _STRATEGY_CASES["grouped-stats-interleaved-chunks"],
    ),
    Sweep(
        "groups_per_axis",
        GROUP_COUNT_AXIS,
        _group_count,
        _STRATEGY_CASES["grouped-stats-group-count"],
    ),
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _raster_size,
        _ROBUST_CASES,
    ),
)


############################
# Report comparisons
############################

_SWEEP_BY_ID = {sweep.id: sweep for sweep in SWEEPS}

COMPARISONS = (
    *tuple(
        comparison(_SWEEP_BY_ID[sweep_id], documentation=False, summary=False)
        for sweep_id in ("grouped-flox-raster-size", "grouped-flox-group-count")
    ),
    comparison(
        _SWEEP_BY_ID["grouped-stats-raster-size"],
        cases=_EXECUTION_CASES,
        slug="grouped-stats-execution-size",
        documentation=False,
    ),
    *tuple(
        comparison(
            _SWEEP_BY_ID[sweep_id],
            cases=_STRATEGY_CASES[sweep_id],
            documentation=False,
        )
        for sweep_id in (
            "grouped-stats-raster-size",
            "grouped-stats-chunk-size",
            "grouped-stats-interleaved-chunks",
            "grouped-stats-group-count",
        )
    ),
    comparison(_SWEEP_BY_ID["grouped-stats-robust-size"], documentation=False),
)
