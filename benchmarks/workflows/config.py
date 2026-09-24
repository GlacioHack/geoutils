"""Define benchmark cases, sweeps, operation hooks and shared runtime settings."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from geoutils.profiler import ProfileMetrics

############################
# Shared names and labels
############################

# Names for modes
ExecutionMode = Literal["eager", "dask", "multiprocessing"]
CalculationEngine = Literal["scipy", "numba", "rasterio", "numpy"]
OperationStrategyName = Literal[
    "sequential", "topk", "label_union", "label_stitch", "geometry_stitch", "auto", "dense", "sparse", "groupwise"
]
OperationName = Literal[
    "crop",
    "translate",
    "copy",
    "clip",
    "filter",
    "reproject",
    "statistics",
    "grouped_stats",
    "subsample",
    "to_pointcloud",
    "interp_points",
    "polygonize",
    "write",
    "rasterize",
    "create_mask",
    "grid",
]
ComparisonDimension = Literal["method", "calculation_engine", "strategy", "execution_mode", "output_format"]
ExternalReference = Literal["gdal_cli", "pdal_cli", "flox"]
PointOutputDriver = Literal["GPKG", "LAS", "LAZ"]
Parameter = int | float

# Labels to print in webpage
GDAL_CLI_LABEL = "GDAL CLI"
PDAL_CLI_LABEL = "PDAL CLI"
FLOX_LABEL = "Flox"
EXECUTION_MODE_LABELS: dict[ExecutionMode, str] = {
    "eager": "Eager",
    "dask": "Dask",
    "multiprocessing": "Multiprocessing",
}
CALCULATION_ENGINE_LABELS: dict[CalculationEngine, str] = {
    "scipy": "SciPy",
    "numba": "Numba",
    "rasterio": "Rasterio/GDAL",
    "numpy": "NumPy",
}
METHOD_LABELS = {
    "nearest": "Nearest",
    "linear": "Linear (Delaunay)",
    "idw": "Inverse-distance",
    "mean": "Circular mean",
}
STRATEGY_LABELS: dict[OperationStrategyName, str] = {
    "sequential": "Sequential",
    "topk": "Top-k",
    "label_union": "Label union",
    "label_stitch": "Label stitch",
    "geometry_stitch": "Geometry stitch",
    "auto": "Automatic",
    "dense": "Dense summaries",
    "sparse": "Sparse summaries",
    "groupwise": "Complete groups",
}
OPERATION_LABELS: dict[OperationName, str] = {
    "crop": "Cropping",
    "translate": "Translation",
    "copy": "Copying",
    "clip": "Clipping",
    "filter": "Filtering",
    "reproject": "Reprojection",
    "statistics": "Statistics",
    "grouped_stats": "Grouped statistics",
    "subsample": "Subsampling",
    "to_pointcloud": "Point cloud conversion",
    "interp_points": "Point interpolation",
    "polygonize": "Polygonization",
    "write": "Writing",
    "rasterize": "Rasterization",
    "create_mask": "Mask creation",
    "grid": "Gridding",
}
PARAMETER_TEXT = {
    "raster_size": ("Size of raster (pixels per side)", "raster size"),
    "chunk_size": ("Size of chunks (pixels per side)", "chunk size"),
    "interpolated_points": ("Number of interpolated points", "point count"),
    "subsample_size": ("Number of output points", "output point count"),
    "points_per_axis": ("Number of source points per axis", "source point count"),
    "groups_per_axis": ("Number of groups per axis", "group count"),
}


@dataclass(frozen=True)
class ParameterAxis:
    """Store one full ASV parameter range and its bounded pull-request value."""

    values: tuple[Parameter, ...]
    pr_value: Parameter

    def parameters(self, pr_check: bool) -> tuple[Parameter, ...]:
        """Return the complete range or its single pull-request value."""

        return (self.pr_value,) if pr_check else self.values


# Use one decimal scale for every raster benchmark and one for raster chunks
RASTER_AXIS = ParameterAxis((2_000, 4_000, 8_000), 1_000)
RASTER_CHUNK_AXIS = ParameterAxis((100, 500, 1_000), 100)
GROUP_COUNT_AXIS = ParameterAxis((10, 20, 50), 10)
POINT_COUNT_AXIS = ParameterAxis((1_000, 10_000, 100_000), 1_000)
GRID_POINT_AXIS = ParameterAxis((10, 20, 40), 10)
POINT_CHUNK_SIZE = 100
WORKER_INTEGRATION_AXIS = ParameterAxis((1_000,), 1_000)
VARIOGRAM_PAIR_AXIS = ParameterAxis((10_000, 100_000, 1_000_000), 1_000)
VARIOGRAM_LAG_AXIS = ParameterAxis((10, 100, 1_000), 10)
DASK_CUTOFF_AXIS = ParameterAxis((262_145, 524_288, 1_048_576), 262_145)


###################################
# Configuration and measurements
###################################


# Keep input sizes, worker settings and measured results consistent across ASV, external CLIs and large-data tests
@dataclass
class BenchmarkConfig:
    """Collect deterministic data, chunk, worker and profiling settings."""

    shape: tuple[int, int] = (2000, 2000)
    chunks: tuple[int, int] = (1000, 1000)
    memory_limit: str = "1GB"
    n_workers: int = 1
    threads_per_worker: int = 1
    gdal_cachemax_mb: int = 64
    profile_interval: float = 0.05
    raster_value: float = 1.0
    subsample_size: int = 2048
    pointcloud_subsample_size: int | None = None
    ninterp: int = 2048
    point_partition_size: int = POINT_CHUNK_SIZE
    polygon_regions_per_axis: int = 1
    vector_features_per_axis: int = 1
    point_features_per_axis: int = 5
    grouped_regions_per_axis: int = 8
    grouped_layout: Literal["local", "interleaved"] = "local"
    operation_method: str | None = None
    calculation_engine: CalculationEngine | None = None
    operation_strategy: OperationStrategyName | None = None
    grid_dist_nodata_pixel: float = float("inf")
    dask_write_batch_size: int = 4
    trim_dask_memory: bool = False
    point_output_driver: PointOutputDriver = "GPKG"
    directory: str | None = None


def process_tree_memory_increase_mb(metrics: ProfileMetrics) -> float:
    """Return peak process-tree memory above its first measured value."""

    if not metrics.process_tree_mem_mb:
        raise RuntimeError("Process-tree memory was not collected for this benchmark result")
    baseline = metrics.process_tree_mem_mb[0][1]
    peak = max(value for _, value in metrics.process_tree_mem_mb)
    return max(0.0, peak - baseline)


class ProfiledResult:
    """Expose complete-process memory measurements shared by all benchmark implementations."""

    metrics: ProfileMetrics

    @property
    def peak_process_tree_mem_mb(self) -> float:
        """Return peak aggregate memory for the measured process and its children."""

        peak = self.metrics.peak_process_tree_mem_mb
        if peak is None:
            raise RuntimeError("Process-tree memory was not collected for this benchmark result")
        return peak

    @property
    def process_tree_mem_increase_mb(self) -> float:
        """Return peak memory above the initialized process-tree baseline."""

        return process_tree_memory_increase_mb(self.metrics)


@dataclass
class BenchmarkResult(ProfiledResult):
    """Store one computed result together with memory and worker-health measurements."""

    value: float
    metrics: ProfileMetrics
    worker_pids_before: tuple[int, ...] | dict[str, int] = field(default_factory=tuple)
    worker_pids_after: tuple[int, ...] | dict[str, int] = field(default_factory=tuple)
    dask_worker_baseline_mem_mb: float | None = None
    output_file: str | None = None

    @property
    def worker_restarted(self) -> bool:
        """Whether the backend replaced a worker during the operation."""

        return self.worker_pids_before != self.worker_pids_after


def _class_token(value: str) -> str:
    """Convert one stable dimension value to part of an ASV class name."""

    return "".join(token.capitalize() for token in value.replace("_", "-").split("-"))


##############################
# Cases, sweeps and reports
##############################


@dataclass(frozen=True)
class BenchmarkCase:
    """Identify one benchmarked method, engine, strategy and execution mode."""

    id: str
    operation: OperationName
    method: str | None = None
    engine: CalculationEngine | None = None
    execution: ExecutionMode = "eager"
    strategy: OperationStrategyName | None = None
    output_driver: PointOutputDriver = "GPKG"
    pr_check: bool = False
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def benchmark_class(self) -> str:
        """Return the stable public ASV class name for this case."""

        values = (
            self.execution,
            None if self.output_driver == "GPKG" else self.output_driver,
            self.method,
            self.engine,
            self.strategy,
            self.id,
        )
        return "".join(_class_token(value) for value in values if value is not None)

    @property
    def comparison_group(self) -> str:
        """Return the compatibility name used by existing report code."""

        return self.id

    @property
    def calculation_engine(self) -> CalculationEngine | None:
        """Return the calculation engine under its existing report field name."""

        return self.engine

    @property
    def execution_mode(self) -> ExecutionMode:
        """Return the execution mode under its existing report field name."""

        return self.execution


@dataclass(frozen=True)
class ExternalReferenceCase:
    """Identify one external reference without treating it as an engine or execution mode."""

    id: str
    operation: OperationName
    method: str | None
    external_reference: ExternalReference
    output_driver: PointOutputDriver = "GPKG"
    pr_check: bool = False
    strategy: None = None
    execution: ExecutionMode | None = None

    @property
    def benchmark_class(self) -> str:
        """Return the stable public ASV class name for this reference."""

        values = (
            self.external_reference,
            None if self.output_driver == "GPKG" else self.output_driver,
            self.execution,
            self.method,
            self.id,
        )
        return "".join(_class_token(value) for value in values if value is not None)

    @property
    def comparison_group(self) -> str:
        """Return the compatibility name used by existing report code."""

        return self.id

    @property
    def execution_mode(self) -> ExecutionMode | None:
        """Return the execution mode under its existing report field name."""

        return self.execution


SweepUpdate = Callable[[Parameter, BenchmarkCase | ExternalReferenceCase, bool], Mapping[str, Any]]


@dataclass(frozen=True)
class Sweep:
    """Describe one numeric parameter axis and the cases measured on it."""

    param_name: str
    axis: ParameterAxis
    update: SweepUpdate
    cases: tuple[BenchmarkCase, ...]
    references: tuple[ExternalReferenceCase, ...] = ()
    base: Mapping[str, Any] = field(default_factory=dict)
    harness: type[Any] | None = None

    def __post_init__(self) -> None:
        """Check that every case identifies the same sweep."""

        identifiers = {case.id for case in (*self.cases, *self.references)}
        if len(identifiers) != 1:
            raise ValueError(f"Expected one sweep ID, got: {sorted(identifiers)}")

    @property
    def id(self) -> str:
        """Return the identifier shared by this sweep's cases."""

        return self.cases[0].id

    @property
    def values(self) -> tuple[Parameter, ...]:
        """Return the full parameter range used by scheduled benchmarks."""

        return self.axis.values

    @property
    def pr_value(self) -> Parameter:
        """Return the bounded parameter used by pull-request checks."""

        return self.axis.pr_value

    def parameters(self, pr_check: bool) -> tuple[Parameter, ...]:
        """Return the full axis or its single pull-request sample."""

        return self.axis.parameters(pr_check)

    def make_config(
        self, parameter: Parameter, case: BenchmarkCase | ExternalReferenceCase, pr_check: bool
    ) -> BenchmarkConfig:
        """Build one runtime configuration from the fixed and parameter-dependent values."""

        values = {**self.base, **self.update(parameter, case, pr_check)}
        return BenchmarkConfig(**values)


@dataclass(frozen=True)
class Comparison:
    """Describe one parameter plot while varying exactly one categorical dimension."""

    slug: str
    title: str
    parameter_label: str
    cases: tuple[BenchmarkCase, ...]
    references: tuple[ExternalReferenceCase, ...]
    operation: OperationName
    method: str | None
    sweep: Sweep
    logarithmic_x: bool = False
    documentation: bool = True
    summary: bool = True
    series_dimension: ComparisonDimension = "execution_mode"
    calculation_engine: CalculationEngine | None = None
    strategy: OperationStrategyName | None = None
    execution_mode: ExecutionMode | None = None

    @property
    def series(self) -> tuple[tuple[str, str], ...]:
        """Return plot labels and ASV classes generated from the selected cases."""

        return _comparison_series(self.cases, self.references, self.series_dimension)

    def workload(self, parameter: Parameter) -> str:
        """Describe the fixture values shared by every series at one parameter value."""

        cases: tuple[BenchmarkCase | ExternalReferenceCase, ...] = (*self.cases, *self.references)
        configs = tuple(self.sweep.make_config(parameter, case, False) for case in cases)

        # Only show values shared by every line so method comparisons do not imply one method's fixture
        def common(name: str) -> Any | None:
            values = [getattr(config, name) for config in configs]
            return values[0] if all(value == values[0] for value in values[1:]) else None

        parts: list[str] = []
        shape = common("shape")
        if shape is not None:
            parts.append(f"{shape[0]:,} × {shape[1]:,} raster")
        chunks = common("chunks")
        if chunks is not None:
            parts.append(f"{chunks[0]:,} × {chunks[1]:,} chunks")

        # Add operation-specific counts that materially define the generated fixture
        if self.operation == "interp_points" and (ninterp := common("ninterp")) is not None:
            parts.append(f"{ninterp:,} interpolated points")
        if self.operation == "subsample" and (subsample_size := common("subsample_size")) is not None:
            parts.append(f"{subsample_size:,} output points")
        if self.operation == "grid" and (points := common("point_features_per_axis")) is not None:
            parts.append(f"{points:,} × {points:,} source points")
            has_dask = any(isinstance(case, BenchmarkCase) and case.execution == "dask" for case in cases)
            if has_dask and (point_chunk := common("point_partition_size")) is not None:
                parts.append(f"{point_chunk:,}-point chunks")
        if self.operation in {"clip", "rasterize"} and (features := common("vector_features_per_axis")) is not None:
            parts.append(f"{features:,} × {features:,} vector features")
        if self.operation == "polygonize" and (regions := common("polygon_regions_per_axis")) is not None:
            parts.append(f"{regions:,} × {regions:,} raster regions")
        if self.operation == "grouped_stats" and (groups := common("grouped_regions_per_axis")) is not None:
            layout = common("grouped_layout")
            parts.append(f"{groups:,} × {groups:,} {layout} groups")
        return "; ".join(parts)


def _comparison_choices(
    cases: tuple[BenchmarkCase, ...],
) -> tuple[
    OperationName,
    str | None,
    CalculationEngine | None,
    OperationStrategyName | None,
    ExecutionMode | None,
    ComparisonDimension,
]:
    """Infer fixed choices and the one choice varied between plotted GeoUtils cases."""

    operations = {case.operation for case in cases}
    if len(operations) != 1:
        raise ValueError(f"Expected one comparison operation, got: {sorted(operations)}")

    dimension = _comparison_dimension(cases)

    def fixed(attribute: str, varied_dimension: ComparisonDimension) -> Any | None:
        if dimension == varied_dimension:
            return None
        values = {getattr(case, attribute) for case in cases if getattr(case, attribute) is not None}
        return next(iter(values)) if len(values) == 1 else None

    return (
        next(iter(operations)),
        fixed("method", "method"),
        fixed("engine", "calculation_engine"),
        fixed("strategy", "strategy"),
        fixed("execution", "execution_mode"),
        dimension,
    )


def _comparison_dimension(cases: tuple[BenchmarkCase, ...]) -> ComparisonDimension:
    """Return the one implementation choice that differs between benchmark cases."""

    # Check choices in an order that treats the eager absence of a chunk strategy as part of execution mode
    dimensions: tuple[tuple[ComparisonDimension, str], ...] = (
        ("output_format", "output_driver"),
        ("calculation_engine", "engine"),
        ("method", "method"),
        ("strategy", "strategy"),
        ("execution_mode", "execution"),
    )
    varied: list[ComparisonDimension] = []
    for dimension, attribute in dimensions:
        values = {getattr(case, attribute) for case in cases if getattr(case, attribute) is not None}
        if len(values) > 1:
            varied.append(dimension)
    if "execution_mode" in varied and "strategy" in varied:
        varied.remove("strategy")
    if len(varied) != 1:
        raise ValueError(f"Expected one varied comparison choice, got: {varied}")
    return varied[0]


def comparison(
    sweep: Sweep,
    *,
    cases: tuple[BenchmarkCase, ...] | None = None,
    references: tuple[ExternalReferenceCase, ...] | None = None,
    slug: str | None = None,
    logarithmic_x: bool = False,
    documentation: bool = True,
    summary: bool = True,
) -> Comparison:
    """Build concise report metadata from a sweep and its varied implementation choice."""

    selected_cases = sweep.cases if cases is None else cases
    selected_references = sweep.references if references is None else references
    operation, method, calculation_engine, strategy, execution_mode, series_dimension = _comparison_choices(
        selected_cases
    )

    # Name the operation and any fixed method before describing the choice represented by separate series
    subject = OPERATION_LABELS[operation]
    if method is not None:
        method_label = METHOD_LABELS.get(method, method.replace("_", " ").title())
        subject = f"{method_label} {subject.lower()}"
    choice = {
        "execution_mode": "",
        "calculation_engine": " engines",
        "method": " methods",
        "strategy": " strategies",
        "output_format": " formats",
    }[series_dimension]
    parameter_label, parameter_title = PARAMETER_TEXT[sweep.param_name]
    generated_title = f"{subject}{choice} by {parameter_title}"

    return Comparison(
        slug=sweep.id if slug is None else slug,
        title=generated_title,
        parameter_label=parameter_label,
        cases=selected_cases,
        references=selected_references,
        operation=operation,
        method=method,
        sweep=sweep,
        logarithmic_x=logarithmic_x,
        documentation=documentation,
        summary=summary,
        series_dimension=series_dimension,
        calculation_engine=calculation_engine,
        strategy=strategy,
        execution_mode=execution_mode,
    )


OperationHandler = Callable[[Any, BenchmarkCase], float]
OptionBuilder = Callable[[BenchmarkCase, BenchmarkConfig], Mapping[str, Any]]


#####################################
# Operation and coverage contracts
#####################################


@dataclass(frozen=True)
class Operation:
    """Connect one operation name with its handler, public options and benchmark defaults."""

    name: OperationName
    handler: OperationHandler
    option_builder: OptionBuilder
    label_options: tuple[str, ...] = ()
    method_engines: Mapping[str | None, tuple[CalculationEngine, ...]] = field(default_factory=dict)
    default_method: str | None = None
    strategies: tuple[OperationStrategyName, ...] = ()
    default_strategy: OperationStrategyName | None = None
    call_name: str | None = None

    @property
    def public_call_name(self) -> str:
        """Return the public call name shown beside benchmark results."""

        return f".{self.name}" if self.call_name is None else self.call_name

    def resolve_case(self, execution: ExecutionMode, config: BenchmarkConfig) -> BenchmarkCase:
        """Resolve and validate one runner request against this operation's local choices."""

        # Select the method and engine locally instead of consulting a package-wide registry
        method = config.operation_method
        engine = config.calculation_engine
        if self.method_engines:
            method = self.default_method if method is None else method
            engines = self.method_engines.get(method)
            if engines is None:
                raise ValueError(f"Expected one benchmark method for {self.name!r}/{method!r}")
            engine = engines[0] if engine is None else engine
            if engine not in engines:
                raise ValueError(f"Engine {engine!r} does not support benchmark method {self.name!r}/{method!r}")
        elif method is not None or engine is not None:
            raise ValueError(f"Operation {self.name!r} has no registered benchmark method or calculation engine")

        # Strategies apply only to chunked operations and use the operation's declared default
        strategy = config.operation_strategy
        if self.strategies:
            if execution == "eager":
                if strategy is not None:
                    raise ValueError(f"Strategy {strategy!r} only applies to chunked execution of {self.name!r}")
                strategy = None
            else:
                strategy = self.default_strategy if strategy is None else strategy
                if strategy not in self.strategies:
                    raise ValueError(f"Expected one benchmark strategy for {self.name!r}/{strategy!r}")
        elif strategy is not None:
            raise ValueError(f"Operation {self.name!r} has no registered benchmark strategy")

        return BenchmarkCase(
            id=self.name,
            operation=self.name,
            method=method,
            engine=engine,
            execution=execution,
            strategy=strategy,
        )


@dataclass(frozen=True)
class OperationCoverage:
    """Describe one operation and the out-of-core execution modes tested for it."""

    operation: OperationName
    execution_modes: tuple[ExecutionMode, ...]
    expected_value: float
    order: int


############################
# Compact case factories
############################


def execution_cases(
    sweep_id: str,
    operation: OperationName,
    method: str | None,
    engine: CalculationEngine | None,
    *,
    strategy: OperationStrategyName | None = None,
    executions: tuple[ExecutionMode, ...] = ("eager", "dask", "multiprocessing"),
    pr_executions: tuple[ExecutionMode, ...] = (),
    output_driver: PointOutputDriver = "GPKG",
) -> tuple[BenchmarkCase, ...]:
    """Generate an execution comparison around fixed numerical choices."""

    return tuple(
        BenchmarkCase(
            sweep_id,
            operation,
            method,
            engine,
            execution,
            strategy if execution != "eager" else None,
            output_driver,
            execution in pr_executions,
        )
        for execution in executions
    )


def strategy_cases(
    sweep_id: str,
    operation: OperationName,
    method: str | None,
    engine: CalculationEngine | None,
    strategies: tuple[OperationStrategyName, ...],
    *,
    execution: Literal["dask", "multiprocessing"],
) -> tuple[BenchmarkCase, ...]:
    """Generate cases for approaches that coordinate one chunked operation."""

    return tuple(BenchmarkCase(sweep_id, operation, method, engine, execution, strategy) for strategy in strategies)


def external_case(
    benchmark: BenchmarkCase | tuple[BenchmarkCase, ...],
    *,
    reference: ExternalReference = "gdal_cli",
    pr_check: bool = False,
    execution: ExecutionMode | None = None,
) -> ExternalReferenceCase:
    """Define one external reference equivalent to a GeoUtils operation."""

    case = benchmark[0] if isinstance(benchmark, tuple) else benchmark
    if isinstance(benchmark, tuple):
        choices = {(item.id, item.operation, item.method, item.output_driver) for item in benchmark}
        if len(choices) != 1:
            raise ValueError(f"Expected equivalent benchmark cases for one external reference, got: {choices}")
    return ExternalReferenceCase(
        case.id,
        case.operation,
        case.method,
        reference,
        case.output_driver,
        pr_check,
        execution=execution,
    )


def merge_cases(*groups: tuple[BenchmarkCase, ...]) -> tuple[BenchmarkCase, ...]:
    """Deduplicate cases reused by several plots while retaining pull-request selection."""

    cases: dict[str, BenchmarkCase] = {}
    for case in (case for group in groups for case in group):
        existing = cases.get(case.benchmark_class)
        if existing is not None and existing != case:
            if (
                replace(existing, pr_check=case.pr_check) != case
                and replace(case, pr_check=existing.pr_check) != existing
            ):
                raise ValueError(f"Conflicting benchmark case ID: {case.benchmark_class}")
            case = replace(existing, pr_check=existing.pr_check or case.pr_check)
        cases[case.benchmark_class] = case
    return tuple(cases.values())


def series_label(case: BenchmarkCase, dimension: ComparisonDimension) -> str:
    """Return the report label for the dimension varied by one case."""

    if dimension == "execution_mode":
        return EXECUTION_MODE_LABELS[case.execution]
    if dimension == "calculation_engine":
        assert case.engine is not None
        return CALCULATION_ENGINE_LABELS[case.engine]
    if dimension == "strategy":
        assert case.strategy is not None
        return STRATEGY_LABELS[case.strategy]
    if dimension == "output_format":
        return case.output_driver
    assert case.method is not None
    return METHOD_LABELS.get(case.method, case.method.replace("_", " ").title())


def _comparison_series(
    cases: tuple[BenchmarkCase, ...],
    references: tuple[ExternalReferenceCase, ...],
    dimension: ComparisonDimension,
) -> tuple[tuple[str, str], ...]:
    """Return labelled GeoUtils and external ASV classes for one plot."""

    series: list[tuple[str, str]] = []
    if dimension == "output_format":
        for case in cases:
            series.append((f"GeoUtils {case.output_driver}", case.benchmark_class))
            for reference in references:
                if reference.output_driver == case.output_driver:
                    label = {"gdal_cli": "GDAL", "pdal_cli": "PDAL", "flox": FLOX_LABEL}[reference.external_reference]
                    series.append((f"{label} {case.output_driver}", reference.benchmark_class))
        return tuple(series)

    series.extend((series_label(case, dimension), case.benchmark_class) for case in cases)
    for reference in references:
        label = {
            "gdal_cli": GDAL_CLI_LABEL,
            "pdal_cli": PDAL_CLI_LABEL,
            "flox": FLOX_LABEL,
        }[reference.external_reference]
        if reference.execution is not None:
            label = f"{label} ({EXECUTION_MODE_LABELS[reference.execution]})"
        series.append((label, reference.benchmark_class))
    return tuple(series)
