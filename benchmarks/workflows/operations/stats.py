"""Define raster and grouped-statistics benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from benchmarks.asv_suite import asv_parameter_values
from benchmarks.comparisons.dask import global_statistics
from benchmarks.comparisons.flox import compute_geoutils_grouped_stats, prepare_grouped_inputs
from benchmarks.comparisons.flox import grouped_stats as flox_grouped_stats
from benchmarks.workflows.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_RASTER_SIZE,
    GROUP_COUNTS,
    RASTER_CHUNK_SIZES,
    RASTER_SIZES,
    Parameter,
    RuntimeConfig,
    raster_size_config,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    merge_cases,
    parameter_config,
    reference_case,
    strategy_cases,
)
from benchmarks.workflows.io import write_constant_raster
from geoutils._misc import import_optional
from geoutils.stats.reduction import (
    _normalize_statistics,
    _reduce_values,
)

ORDER = 10
STRATEGIES = ("auto", "dense", "sparse", "groupwise")
STRATEGY_LABELS = {
    "auto": "Automatic",
    "dense": "Dense summaries",
    "sparse": "Sparse summaries",
    "groupwise": "Complete groups",
}

##########################################
# Define setup for statistics operations
##########################################


def statistics_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Build the public options for regular or grouped statistics."""

    if config.value("operation") == "statistics":
        return {"statistics": ["mean", "std", "valid count"]}
    statistics = ["mean", "std", "min", "max"] if case.method == "moments" else ["median", "nmad"]
    return {"statistics": statistics, "strategy": case.strategy or "auto"}


def prepare_statistics(runner: Any, case: Case) -> None:
    """Prepare the raster or shared arrays used by one statistics case."""

    if case.variant == "flox" or case.implementation == "flox":
        if case.implementation == "flox":
            try:
                import_optional("flox")
            except ImportError as exc:
                raise NotImplementedError("Install optional flox to run this comparison") from exc
        runner.grouped_inputs = prepare_grouped_inputs(
            runner.config.shape[0],
            int(runner.config.value("grouped_regions_per_axis", 8)),
            runner.config.value("grouped_layout", "local"),
            runner.backend,
            runner.config.chunks,
        )
        return
    write_constant_raster(runner.path("source-raster.tif"), runner.config)


def run_statistics(runner: Any, case: Case) -> float:
    """Compute regular reductions or grouped statistics on the prepared raster."""

    if case.variant == "flox" or case.implementation == "flox":
        if case.implementation == "flox":
            flox_grouped_stats(*runner.grouped_inputs, use_dask=runner.backend == "dask")
        else:
            if runner.backend == "multiprocessing":
                from geoutils.multiproc import MultiprocConfig

                mp_config = MultiprocConfig(chunks=runner.config.chunks, cluster=runner.mp_cluster)
            else:
                mp_config = None
            compute_geoutils_grouped_stats(*runner.grouped_inputs, mp_config=mp_config)
        return 1.0

    raster = runner.make_raster()
    if runner.operation.name == "statistics":
        if runner.backend != "dask":
            raise ValueError("Raster statistics are only registered for Dask")

        # Compute selected reductions without evaluating unrelated quantiles
        import_optional("dask", extra_name="benchmark")
        import dask

        statistics = raster.rst.stats(statistics_options(case, runner.config)["statistics"])
        mean, _, _ = dask.compute(*statistics.values())
        return float(mean)

    return grouped_statistics(runner, raster, case)


def grouped_statistics(runner: Any, raster: Any, case: Case) -> float:
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
    regions = int(runner.config.value("grouped_regions_per_axis", 8))
    if regions < 1 or regions > min(height, width):
        raise ValueError("Grouped regions per axis must fit within the raster dimensions.")

    # Separate localized membership from groups repeated through every chunk
    if runner.config.value("grouped_layout", "local") == "local":
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
    mp_config = runner._multiproc_config() if runner.backend == "multiprocessing" else None
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
STATISTICS = Operation(
    "statistics",
    prepare_statistics,
    run_statistics,
    statistics_options,
    call_name=".stats",
    label="Statistics",
    benchmark_name="stats",
    order=6,
    large_data_cases=(Case(execution="dask", options={"operation": "statistics"}),),
)
GROUPED_STATS = Operation(
    "grouped_stats",
    prepare_statistics,
    run_statistics,
    statistics_options,
    ("statistics", "strategy"),
    call_name="stats",
    label="Grouped statistics",
    benchmark_name="stats",
    order=7,
    large_data_cases=(
        Case(
            method="moments",
            engine="numpy",
            execution="dask",
            strategy="auto",
            options={"operation": "grouped_stats"},
        ),
    ),
)
OPERATIONS = (STATISTICS, GROUPED_STATS)


###################
# Define benchmarks
###################


# Each case fixes one method, strategy and execution mode for one ASV result series; its sweep owns the changing input
# The sweeps isolate input size, chunk size, membership layout and group count for shared grouped-statistic kernels
EXECUTION_CASES = execution_cases(
    "moments",
    "numpy",
    strategy="dense",
    options={"operation": "grouped_stats"},
    labels={"engine": "NumPy", "strategy": STRATEGY_LABELS["dense"]},
)
STRATEGY_CASES = {
    sweep_id: strategy_cases(
        "moments",
        "numpy",
        STRATEGIES,
        execution="dask",
        options={"operation": "grouped_stats"},
        strategy_labels=STRATEGY_LABELS,
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
    "robust",
    "numpy",
    strategy="groupwise",
    options={"operation": "grouped_stats"},
    labels={"engine": "NumPy", "strategy": STRATEGY_LABELS["groupwise"]},
)

# Compare public GeoUtils stats() with direct Flox reductions on the same prepared arrays
FLOX_CASES = {
    sweep_id: execution_cases(
        "moments",
        "numpy",
        strategy="auto",
        options={"operation": "grouped_stats"},
        labels={"engine": "NumPy", "strategy": STRATEGY_LABELS["auto"]},
        variant="flox",
    )
    for sweep_id in ("grouped-flox-raster-size", "grouped-flox-group-count")
}
FLOX_REFERENCES = {
    sweep_id: tuple(
        reference_case(
            FLOX_CASES[sweep_id],
            implementation="flox",
            execution=execution,
        )
        for execution in ("inmem", "dask")
    )
    for sweep_id in FLOX_CASES
}


def chunk_size(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Include uneven edge chunks and groups crossing partition boundaries."""

    assert parameter is not None
    chunk_size = int(parameter)
    return {"shape": (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE), "chunks": (chunk_size, chunk_size)}


def interleaved_chunk_size(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Keep observations interleaved across all chunks for each tested partition size."""

    return {**chunk_size(parameter, case), "grouped_layout": "interleaved"}


def group_count(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Include 4225 groups so automatic reduction exercises its sparse branch."""

    assert parameter is not None
    return {
        "shape": (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE),
        "chunks": (500, 500),
        "grouped_regions_per_axis": int(parameter),
    }


def flox_raster_size(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Vary raster size around 256 local groups and fixed spatial chunks."""

    assert parameter is not None
    size = int(parameter)
    return {
        "shape": (size, size),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "grouped_regions_per_axis": 20,
    }


def flox_group_count(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Vary declared groups on a fixed raster with membership repeated across chunks."""

    assert parameter is not None
    return {
        "shape": (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "grouped_regions_per_axis": int(parameter),
        "grouped_layout": "interleaved",
    }


def grouped_workload(parameter: Parameter, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the raster, chunks and grouped membership layout."""

    config = configs[0]
    regions = int(config.value("grouped_regions_per_axis", 8))
    layout = config.value("grouped_layout", "local")
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} raster; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; "
        f"{regions:,} × {regions:,} {layout} groups"
    )


# Measure the selected cases over each numeric axis, including matched Flox cases for the first two sweeps
BENCHMARKS = (
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        GROUPED_STATS,
        (*FLOX_CASES["grouped-flox-raster-size"], *FLOX_REFERENCES["grouped-flox-raster-size"]),
        flox_raster_size,
        name="grouped-flox",
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "groups_per_axis",
        GROUP_COUNTS,
        GROUPED_STATS,
        (*FLOX_CASES["grouped-flox-group-count"], *FLOX_REFERENCES["grouped-flox-group-count"]),
        flox_group_count,
        name="grouped-flox",
        parameter_label="Number of groups per axis",
        parameter_title="group count",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        GROUPED_STATS,
        merge_cases(EXECUTION_CASES, STRATEGY_CASES["grouped-stats-raster-size"]),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "chunk_size",
        RASTER_CHUNK_SIZES,
        GROUPED_STATS,
        STRATEGY_CASES["grouped-stats-chunk-size"],
        chunk_size,
        parameter_label="Size of chunks (pixels per side)",
        parameter_title="chunk size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "chunk_size",
        RASTER_CHUNK_SIZES,
        GROUPED_STATS,
        STRATEGY_CASES["grouped-stats-interleaved-chunks"],
        interleaved_chunk_size,
        name="grouped-stats-interleaved",
        parameter_label="Size of chunks (pixels per side)",
        parameter_title="chunk size",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "groups_per_axis",
        GROUP_COUNTS,
        GROUPED_STATS,
        STRATEGY_CASES["grouped-stats-group-count"],
        group_count,
        parameter_label="Number of groups per axis",
        parameter_title="group count",
        describe_workload=grouped_workload,
    ),
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        GROUPED_STATS,
        ROBUST_CASES,
        raster_size_config,
        name="grouped-stats-robust",
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
        describe_workload=grouped_workload,
    ),
)


#############################
# Define report comparisons
#############################

# Comparisons select saved GeoUtils and Flox series for external plots, then GeoUtils series for internal strategy plots
COMPARISONS = (
    *tuple(comparison(benchmark, by="execution", documentation=False, summary=False) for benchmark in BENCHMARKS[:2]),
    comparison(
        BENCHMARKS[2],
        by="execution",
        cases=EXECUTION_CASES,
        slug="grouped-stats-execution-size",
        documentation=False,
    ),
    *tuple(
        comparison(
            benchmark,
            by="strategy",
            cases=STRATEGY_CASES[sweep_id],
            documentation=False,
        )
        for benchmark, sweep_id in zip(
            BENCHMARKS[2:6],
            (
                "grouped-stats-raster-size",
                "grouped-stats-chunk-size",
                "grouped-stats-interleaved-chunks",
                "grouped-stats-group-count",
            ),
            strict=True,
        )
    ),
    comparison(BENCHMARKS[6], by="execution", documentation=False),
)


###################################
# Internal reduction microbenchmark
###################################


class GlobalDaskReduction:
    """Measure the shared GeoUtils reducer and native Dask reductions on the same prepared array."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    timeout = 300
    param_names = ["implementation", "raster_size"]
    params = [["geoutils", "native_dask"], asv_parameter_values(RASTER_SIZES)]

    def setup(self, implementation: Literal["geoutils", "native_dask"], raster_size: int) -> None:
        """Prepare one finite Dask raster and the same mergeable statistic request for both implementations."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        del implementation
        generator = np.random.default_rng(42)
        values = generator.normal(size=(raster_size, raster_size))
        values[::97, ::89] = np.nan
        self.values = da.from_array(values, chunks=(500, 500))
        self.aliases = {"mean", "min", "max", "sum", "sumofsquares", "rmse", "std"}
        self.statistics = _normalize_statistics(sorted(self.aliases), grouped=False)

    def time_reduction(self, implementation: Literal["geoutils", "native_dask"], raster_size: int) -> None:
        """Compute the requested global estimates with the selected Dask reduction implementation."""

        import dask

        del raster_size
        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils":
                _reduce_values([self.values], self.statistics)
            else:
                global_statistics(self.values)


# Keep the stored identifier while locating this internal comparison outside the public ASV modules
setattr(
    GlobalDaskReduction.time_reduction,
    "benchmark_name",
    "asv_suite.global_stats.GlobalDaskReduction.time_reduction",
)
