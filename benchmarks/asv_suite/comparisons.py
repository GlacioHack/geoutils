"""Generate ASV cases that measure time and RAM while varying one benchmark dimension at a time."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, replace
from typing import Literal, cast

from benchmarks.asv_suite import asv_parameter_values, asv_pr_check_enabled
from benchmarks.gdal_comparison.commands import ComparisonOperation
from benchmarks.gdal_comparison.runner import GdalRunner
from benchmarks.workflows.registry import (
    OPERATION_METHODS,
    OPERATION_STRATEGIES,
    CalculationEngine,
    ExecutionMode,
    OperationName,
    OperationStrategyName,
)
from benchmarks.workflows.runner import BenchmarkConfig, BenchmarkRunner

# Comparisons vary one GeoUtils choice at a time: method, calculation engine, chunk strategy or execution mode
# The label dictionaries give the stored values readable names in plots
ComparisonDimension = Literal["method", "calculation_engine", "strategy", "execution_mode"]
ExternalReference = Literal["gdal_cli"]
GDAL_CLI_LABEL = "GDAL CLI"

EXECUTION_MODE_LABELS: dict[ExecutionMode, str] = {
    "eager": "Eager",
    "dask": "Dask",
    "multiprocessing": "Multiprocessing",
}
CALCULATION_ENGINE_LABELS: dict[CalculationEngine, str] = {
    "scipy": "SciPy",
    "numba": "Numba",
    "rasterio": "Rasterio/GDAL",
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
}


def _class_token(value: str) -> str:
    """Convert one stable dimension value to part of an ASV class name."""

    return "".join(token.capitalize() for token in value.replace("_", "-").split("-"))


# Store one concrete combination, such as eager IDW gridding with Numba, before creating its ASV class
@dataclass(frozen=True)
class BenchmarkCase:
    """Identify one valid GeoUtils method, engine, strategy and execution-mode case."""

    comparison_group: str
    operation: OperationName
    method: str | None
    calculation_engine: CalculationEngine | None
    strategy: OperationStrategyName | None
    execution_mode: ExecutionMode
    pr_check: bool = False

    @property
    def benchmark_class(self) -> str:
        """Return the generated public ASV class name for this case."""

        values = (
            self.execution_mode,
            self.method,
            self.calculation_engine,
            self.strategy,
            self.comparison_group,
        )
        return "".join(_class_token(value) for value in values if value is not None)


@dataclass(frozen=True)
class ExternalReferenceCase:
    """Identify one external reference without treating it as an engine or execution mode."""

    comparison_group: str
    operation: OperationName
    method: str | None
    external_reference: ExternalReference
    pr_check: bool = False
    strategy: None = None

    @property
    def benchmark_class(self) -> str:
        """Return the generated public ASV class name for this reference."""

        values = (self.external_reference, self.method, self.comparison_group)
        return "".join(_class_token(value) for value in values if value is not None)


def _execution_cases(
    comparison_group: str,
    operation: OperationName,
    method: str | None,
    calculation_engine: CalculationEngine | None,
    *,
    strategy: OperationStrategyName | None = None,
    execution_modes: tuple[ExecutionMode, ...] = ("eager", "dask", "multiprocessing"),
    pr_modes: tuple[ExecutionMode, ...] = (),
) -> tuple[BenchmarkCase, ...]:
    """Generate an execution-mode comparison around fixed numerical dimensions."""

    return tuple(
        BenchmarkCase(
            comparison_group,
            operation,
            method,
            calculation_engine,
            strategy if execution_mode != "eager" else None,
            execution_mode,
            pr_check=execution_mode in pr_modes,
        )
        for execution_mode in execution_modes
    )


def _engine_cases(
    comparison_group: str,
    operation: OperationName,
    method: str | None,
    *,
    execution_mode: ExecutionMode = "eager",
    strategy: OperationStrategyName | None = None,
    pr_engines: tuple[CalculationEngine, ...] = (),
) -> tuple[BenchmarkCase, ...]:
    """Generate an engine comparison for one method and execution mode."""

    if execution_mode == "eager" and strategy is not None:
        raise ValueError("Chunk strategies cannot be fixed for an eager engine comparison")
    specification = next(item for item in OPERATION_METHODS if item.operation == operation and item.method == method)
    return tuple(
        BenchmarkCase(
            comparison_group,
            operation,
            method,
            calculation_engine,
            strategy,
            execution_mode,
            pr_check=calculation_engine in pr_engines,
        )
        for calculation_engine in specification.calculation_engines
    )


def _method_cases(
    comparison_group: str,
    operation: OperationName,
    methods: tuple[str, ...],
    calculation_engine: CalculationEngine,
    *,
    execution_mode: ExecutionMode = "eager",
    strategy: OperationStrategyName | None = None,
) -> tuple[BenchmarkCase, ...]:
    """Generate a method comparison for one engine, strategy and execution mode."""

    if execution_mode == "eager" and strategy is not None:
        raise ValueError("Chunk strategies cannot be fixed for an eager method comparison")
    supported = {item.method: item.calculation_engines for item in OPERATION_METHODS if item.operation == operation}
    if any(calculation_engine not in supported.get(method, ()) for method in methods):
        raise ValueError(f"Engine {calculation_engine!r} does not support every requested {operation!r} method")
    return tuple(
        BenchmarkCase(
            comparison_group,
            operation,
            method,
            calculation_engine,
            strategy,
            execution_mode,
        )
        for method in methods
    )


def _strategy_cases(
    comparison_group: str,
    operation: OperationName,
    method: str | None,
    calculation_engine: CalculationEngine | None,
    *,
    execution_mode: Literal["dask", "multiprocessing"],
) -> tuple[BenchmarkCase, ...]:
    """Generate a comparison of approaches for coordinating one chunked operation."""

    strategies = tuple(item.strategy for item in OPERATION_STRATEGIES if item.operation == operation)
    return tuple(
        BenchmarkCase(
            comparison_group,
            operation,
            method,
            calculation_engine,
            strategy,
            execution_mode,
        )
        for strategy in strategies
    )


def _merge_cases(*groups: tuple[BenchmarkCase, ...]) -> tuple[BenchmarkCase, ...]:
    """Deduplicate cases reused by several plots while retaining pull-request selection."""

    cases: dict[tuple[object, ...], BenchmarkCase] = {}
    for group in groups:
        for case in group:
            key = (
                case.comparison_group,
                case.operation,
                case.method,
                case.calculation_engine,
                case.strategy,
                case.execution_mode,
            )
            existing = cases.get(key)
            cases[key] = replace(case, pr_check=True) if existing is not None and case.pr_check else existing or case
    return tuple(cases.values())


def _external_case(
    comparison_group: str,
    operation: OperationName,
    method: str | None,
    *,
    pr_check: bool = False,
) -> ExternalReferenceCase:
    """Define one GDAL CLI reference equivalent to a GeoUtils operation."""

    return ExternalReferenceCase(comparison_group, operation, method, "gdal_cli", pr_check=pr_check)


# Define the cases needed to compare each operation across execution modes, calculation engines, methods or strategies
# Each helper changes only that choice and keeps the other operation settings fixed
_INTERPOLATION_MODES = _execution_cases("interpolation-point-count", "interp_points", "linear", "scipy")
_REPROJECTION_MODES = _execution_cases("reprojection-raster-size", "reproject", "nearest", "rasterio")
_FILTER_MODES = _execution_cases(
    "filter-chunk-size",
    "filter",
    "mean",
    "scipy",
    execution_modes=("dask", "multiprocessing"),
)
_POLYGONIZATION_MODES = _execution_cases(
    "polygonization-raster-size", "polygonize", None, "rasterio", strategy="label_stitch"
)
_POLYGONIZATION_STRATEGIES = _strategy_cases(
    "polygonization-raster-size", "polygonize", None, "rasterio", execution_mode="dask"
)
_RASTERIZATION_MODES = _execution_cases("rasterization-raster-size", "rasterize", None, "rasterio")
_SUBSAMPLE_STRATEGIES = _strategy_cases("subsample-size", "subsample", None, None, execution_mode="dask")

# Compare all four gridding methods across execution modes while keeping SciPy as the calculation engine
_GRID_METHODS = ("nearest", "linear", "idw", "mean")
_GRID_MODE_CASES = {
    method: _execution_cases(
        "gridding-raster-size",
        "grid",
        method,
        "scipy",
        pr_modes=("eager", "dask", "multiprocessing") if method == "nearest" else (),
    )
    for method in _GRID_METHODS
}

# Reuse the eager SciPy cases in one plot that isolates the choice of gridding method
_GRID_METHOD_CASES = _method_cases("gridding-raster-size", "grid", _GRID_METHODS, "scipy")

# Compare SciPy and Numba in eager mode for the methods supported by both calculation engines
_GRID_ENGINE_CASES = {
    method: _engine_cases(
        "gridding-raster-size",
        "grid",
        method,
        pr_engines=("numba",) if method == "nearest" else (),
    )
    for method in ("nearest", "idw", "mean")
}

# Repeat the nearest engine comparison while varying source point count instead of raster size
_GRID_POINT_ENGINE_CASES = _engine_cases("gridding-point-count", "grid", "nearest")

# Add one fixed-size run per Numba method and worker execution mode to check that compiled kernels work there
# The eager engine comparisons already measure how these methods scale with raster size
_WORKER_EXECUTION_MODES: tuple[ExecutionMode, ...] = ("dask", "multiprocessing")
_NUMBA_WORKER_CASES = tuple(
    BenchmarkCase(
        "worker-integration",
        "grid",
        method,
        "numba",
        None,
        execution_mode,
        pr_check=(method, execution_mode) in (("idw", "dask"), ("mean", "multiprocessing")),
    )
    for method in ("nearest", "idw", "mean")
    for execution_mode in _WORKER_EXECUTION_MODES
)

# Combine every GeoUtils case and remove duplicates when the same combination appears in several comparisons
BENCHMARK_CASES = _merge_cases(
    _INTERPOLATION_MODES,
    _REPROJECTION_MODES,
    _FILTER_MODES,
    _POLYGONIZATION_MODES,
    _POLYGONIZATION_STRATEGIES,
    _RASTERIZATION_MODES,
    _SUBSAMPLE_STRATEGIES,
    *tuple(_GRID_MODE_CASES.values()),
    _GRID_METHOD_CASES,
    *tuple(_GRID_ENGINE_CASES.values()),
    _GRID_POINT_ENGINE_CASES,
    _NUMBA_WORKER_CASES,
)

# Define matching GDAL CLI runs for operations that have a direct external reference
_REPROJECTION_REFERENCE = _external_case("reprojection-raster-size", "reproject", "nearest")
_POLYGONIZATION_REFERENCE = _external_case("polygonization-raster-size", "polygonize", None)
_RASTERIZATION_REFERENCE = _external_case("rasterization-raster-size", "rasterize", None)
_GRID_REFERENCES = {
    method: _external_case(
        "gridding-raster-size",
        "grid",
        method,
        pr_check=method == "nearest",
    )
    for method in _GRID_METHODS
}
_GRID_POINT_REFERENCE = _external_case("gridding-point-count", "grid", "nearest")

# Collect GDAL runs separately because the CLI is neither a GeoUtils engine nor an execution mode
EXTERNAL_REFERENCE_CASES = (
    _REPROJECTION_REFERENCE,
    _POLYGONIZATION_REFERENCE,
    _RASTERIZATION_REFERENCE,
    *_GRID_REFERENCES.values(),
    _GRID_POINT_REFERENCE,
)

# Map each generated ASV class name back to the operation settings needed during setup
BENCHMARK_CASE_BY_CLASS = {case.benchmark_class: case for case in BENCHMARK_CASES}
EXTERNAL_REFERENCE_CASE_BY_CLASS = {case.benchmark_class: case for case in EXTERNAL_REFERENCE_CASES}


def _series_label(case: BenchmarkCase, dimension: ComparisonDimension) -> str:
    """Return the plot label for the dimension varied by one GeoUtils case."""

    if dimension == "execution_mode":
        return EXECUTION_MODE_LABELS[case.execution_mode]
    if dimension == "calculation_engine":
        assert case.calculation_engine is not None
        return CALCULATION_ENGINE_LABELS[case.calculation_engine]
    if dimension == "strategy":
        assert case.strategy is not None
        return STRATEGY_LABELS[case.strategy]
    assert case.method is not None
    return METHOD_LABELS.get(case.method, case.method.replace("_", " ").title())


def _comparison_series(
    cases: tuple[BenchmarkCase, ...],
    dimension: ComparisonDimension,
    external_reference: ExternalReferenceCase | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return labelled ASV classes for one plot, optionally followed by the GDAL CLI."""

    series = tuple((_series_label(case, dimension), case.benchmark_class) for case in cases)
    if external_reference is not None:
        return (*series, (GDAL_CLI_LABEL, external_reference.benchmark_class))
    return series


@dataclass(frozen=True)
class Comparison:
    """Describe one parameter plot while varying exactly one categorical dimension."""

    slug: str
    title: str
    description: str
    parameter_label: str
    series: tuple[tuple[str, str], ...]
    operation: OperationName
    method: str | None
    workload_template: str
    logarithmic_x: bool = False
    documentation: bool = True
    series_dimension: ComparisonDimension = "execution_mode"
    calculation_engine: CalculationEngine | None = None
    strategy: OperationStrategyName | None = None
    execution_mode: ExecutionMode | None = None


# Concisely identify the shared input and support chosen for each gridding method
_GRID_FIXTURE_DESCRIPTIONS = {
    "nearest": "Grids a 17 × 17 regular WGS84 point set with unlimited nearest-neighbor support onto a square WGS84 raster.",
    "linear": "Grids a 17 × 17 regular WGS84 point set with Delaunay linear interpolation onto a square WGS84 raster.",
    "idw": "Grids a 17 × 17 regular WGS84 point set with inverse-distance weighting and 16-pixel support onto a square WGS84 raster.",
    "mean": "Grids a 17 × 17 regular WGS84 point set with a circular mean and 16-pixel support onto a square WGS84 raster.",
}
_GRID_POINTS_PER_AXIS = {"nearest": 17, "linear": 17, "idw": 17, "mean": 17}


# Define the report plots, including their displayed series and the operation settings held fixed
COMPARISONS: tuple[Comparison, ...] = (
    Comparison(
        slug="interpolation-point-count",
        title="Linear interpolation by number of points (SciPy engine)",
        description=("Interpolates deterministic WGS84 points from a 2048 × 2048 WGS84 raster with 512 × 512 chunks."),
        parameter_label="Number of interpolated points",
        series=_comparison_series(_INTERPOLATION_MODES, "execution_mode"),
        operation="interp_points",
        method="linear",
        workload_template=("2,048 × 2,048 source raster; {parameter} interpolated points; 512 × 512 chunks"),
        calculation_engine="scipy",
        logarithmic_x=True,
    ),
    Comparison(
        slug="reprojection-raster-size",
        title="Nearest reprojection by raster size (Rasterio/GDAL engine)",
        description=(
            "Reprojects a WGS84 (EPSG:4326) raster to UTM zone 32N (EPSG:32632) with nearest-neighbor "
            "resampling while preserving the selected output dimensions."
        ),
        parameter_label="Size of raster (pixels per side)",
        series=_comparison_series(_REPROJECTION_MODES, "execution_mode", _REPROJECTION_REFERENCE),
        operation="reproject",
        method="nearest",
        workload_template="{parameter} × {parameter} input/output raster; 512 × 512 chunks",
        calculation_engine="rasterio",
    ),
    Comparison(
        slug="filter-chunk-size",
        title="Mean filter by chunk size (SciPy engine)",
        description="Applies a 5 × 5 mean filter to a 2048 × 2048 WGS84 raster while varying square chunk size.",
        parameter_label="Size of chunks (pixels per side)",
        series=_comparison_series(_FILTER_MODES, "execution_mode"),
        operation="filter",
        method="mean",
        workload_template="2,048 × 2,048 raster; {parameter} × {parameter} chunks; 5 × 5 filter",
        calculation_engine="scipy",
    ),
    Comparison(
        slug="polygonization-raster-size",
        title="Label-stitch polygonization by raster size (Rasterio/GDAL engine)",
        description=(
            "Polygonizes value-1 pixels in a WGS84 raster containing 21 × 21 disconnected rectangles while "
            "varying raster size."
        ),
        parameter_label="Size of raster (pixels per side)",
        series=_comparison_series(_POLYGONIZATION_MODES, "execution_mode", _POLYGONIZATION_REFERENCE),
        operation="polygonize",
        method=None,
        workload_template=("{parameter} × {parameter} raster; 441 disconnected raster regions; 512 × 512 chunks"),
        calculation_engine="rasterio",
        strategy="label_stitch",
    ),
    Comparison(
        slug="polygonization-strategy-raster-size",
        title="Polygonization chunk strategy (Dask execution)",
        description=(
            "Polygonizes the same 21 × 21 disconnected rectangles with Dask while comparing how polygons "
            "crossing chunk boundaries are reconciled."
        ),
        parameter_label="Size of raster (pixels per side)",
        series=_comparison_series(_POLYGONIZATION_STRATEGIES, "strategy"),
        operation="polygonize",
        method=None,
        workload_template=("{parameter} × {parameter} raster; 441 disconnected raster regions; 512 × 512 chunks"),
        calculation_engine="rasterio",
        execution_mode="dask",
        series_dimension="strategy",
        documentation=False,
    ),
    Comparison(
        slug="rasterization-raster-size",
        title="Rasterization by raster size (Rasterio/GDAL engine)",
        description=("Burns 51 × 51 regularly spaced WGS84 polygons as value 1 into a byte raster with background 0."),
        parameter_label="Size of raster (pixels per side)",
        series=_comparison_series(_RASTERIZATION_MODES, "execution_mode", _RASTERIZATION_REFERENCE),
        operation="rasterize",
        method=None,
        workload_template=("2,601 source polygon features; {parameter} × {parameter} output raster; 512 × 512 chunks"),
        calculation_engine="rasterio",
    ),
    Comparison(
        slug="subsampling-strategy-size",
        title="Subsampling chunk strategy (Dask execution)",
        description=(
            "Selects values with random seed 42 from a 2048 × 2048 WGS84 raster while comparing chunk strategies."
        ),
        parameter_label="Number of sampled values",
        series=_comparison_series(_SUBSAMPLE_STRATEGIES, "strategy"),
        operation="subsample",
        method=None,
        workload_template=("2,048 × 2,048 source raster; {parameter} sampled values; 512 × 512 chunks"),
        execution_mode="dask",
        series_dimension="strategy",
        logarithmic_x=True,
        documentation=False,
    ),
    *tuple(
        Comparison(
            slug="gridding-raster-size" if method == "nearest" else f"{method}-gridding-raster-size",
            title=f"{METHOD_LABELS[method]} gridding execution mode (SciPy engine)",
            description=_GRID_FIXTURE_DESCRIPTIONS[method],
            parameter_label="Size of raster (pixels per side)",
            series=_comparison_series(_GRID_MODE_CASES[method], "execution_mode", _GRID_REFERENCES[method]),
            operation="grid",
            method=method,
            workload_template=(
                f"{{parameter}} × {{parameter}} output raster; "
                f"{_GRID_POINTS_PER_AXIS[method]} × {_GRID_POINTS_PER_AXIS[method]} source points; "
                "512 × 512 chunks"
            ),
            calculation_engine="scipy",
            documentation=method == "nearest",
        )
        for method in _GRID_METHODS
    ),
    Comparison(
        slug="gridding-method-raster-size",
        title="Gridding method (SciPy engine, eager execution)",
        description=(
            "Grids regular WGS84 point sets onto square WGS84 rasters using the fixture and support selected for "
            "each numerical method."
        ),
        parameter_label="Size of raster (pixels per side)",
        series=_comparison_series(_GRID_METHOD_CASES, "method"),
        operation="grid",
        method=None,
        workload_template=(
            "{parameter} × {parameter} output raster; method-specific source point set; 512 × 512 chunks"
        ),
        calculation_engine="scipy",
        execution_mode="eager",
        series_dimension="method",
        documentation=False,
    ),
    *tuple(
        Comparison(
            slug=f"{method}-gridding-engine-raster-size",
            title=f"{METHOD_LABELS[method]} gridding calculation engine (eager execution)",
            description=_GRID_FIXTURE_DESCRIPTIONS[method],
            parameter_label="Size of raster (pixels per side)",
            series=_comparison_series(_GRID_ENGINE_CASES[method], "calculation_engine", _GRID_REFERENCES[method]),
            operation="grid",
            method=method,
            workload_template=(
                f"{{parameter}} × {{parameter}} output raster; "
                f"{_GRID_POINTS_PER_AXIS[method]} × {_GRID_POINTS_PER_AXIS[method]} source points; "
                "512 × 512 chunks"
            ),
            execution_mode="eager",
            series_dimension="calculation_engine",
            documentation=False,
        )
        for method in ("nearest", "idw", "mean")
    ),
    Comparison(
        slug="nearest-gridding-engine-point-count",
        title="Nearest gridding by number of source points (eager execution)",
        description=(
            "Grids a regular WGS84 point set with unlimited nearest-neighbor support onto a fixed 1024 × 1024 "
            "WGS84 raster."
        ),
        parameter_label="Number of source points per axis",
        series=_comparison_series(_GRID_POINT_ENGINE_CASES, "calculation_engine", _GRID_POINT_REFERENCE),
        operation="grid",
        method="nearest",
        workload_template=("1,024 × 1,024 output raster; {parameter} × {parameter} source points; 512 × 512 chunks"),
        execution_mode="eager",
        series_dimension="calculation_engine",
        documentation=False,
    ),
)


# The classes below define which numeric input changes, such as raster size, chunk size or point count
# Generated subclasses later combine that input axis with one concrete operation configuration
class _ComparisonBenchmark:
    """Share ASV settings, dimensional metadata and complete result computation."""

    timeout = 900
    number = 1
    repeat = 2
    rounds = 1
    warmup_time = 0
    operation: OperationName
    operation_method: str | None
    calculation_engine: CalculationEngine | None
    operation_strategy: OperationStrategyName | None
    execution_mode: ExecutionMode | None
    external_reference: ExternalReference | None

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Build the fixed configuration around one selected numeric parameter."""

        raise NotImplementedError

    def setup(self, parameter: int) -> None:
        """Prepare deterministic files and initialize one execution case."""

        benchmark_class = type(self).__name__
        case = BENCHMARK_CASE_BY_CLASS.get(benchmark_class)
        reference_case = EXTERNAL_REFERENCE_CASE_BY_CLASS.get(benchmark_class)
        if (case is None) == (reference_case is None):
            raise ValueError(f"Expected exactly one registered benchmark case for {benchmark_class}")

        selected_case = case or reference_case
        assert selected_case is not None
        if asv_pr_check_enabled() and not selected_case.pr_check:
            raise NotImplementedError("Benchmark case omitted from the pull-request sample")

        # Strategies only identify how Dask or multiprocessing coordinates chunks
        self.operation = selected_case.operation
        self.operation_method = selected_case.method
        self.operation_strategy = selected_case.strategy
        self.calculation_engine = case.calculation_engine if case is not None else None
        self.execution_mode = case.execution_mode if case is not None else None
        self.external_reference = reference_case.external_reference if reference_case is not None else None

        # Input generation remains outside all three measured boundaries
        self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-comparison-")
        self.config = self.make_config(parameter)
        self.config.operation_method = self.operation_method
        self.config.calculation_engine = self.calculation_engine
        self.config.operation_strategy = self.operation_strategy
        self.config.directory = self._tmpdir.name
        self.sources = BenchmarkRunner("eager", self.config).prepare_sources()

        if self.external_reference is not None:
            operation = cast(ComparisonOperation, self.operation)
            self.runner: BenchmarkRunner | GdalRunner = GdalRunner(operation, self.config, self.sources)
        else:
            assert self.execution_mode is not None
            self.runner = BenchmarkRunner(self.execution_mode, self.config).start()

    def teardown(self, parameter: int) -> None:
        """Stop workers and remove generated source, output and spill files."""

        if not hasattr(self, "runner"):
            return
        self.runner.close()
        if self.sources is not self.runner:
            self.sources.close()
        self._tmpdir.cleanup()

    def time_operation(self, parameter: int) -> None:
        """Measure a complete operation after execution-mode initialization."""

        if isinstance(self.runner, GdalRunner):
            self.runner._execute()
        else:
            self.runner._execute(self.operation)

    def track_end_to_end_time_s(self, parameter: int) -> float:
        """Measure execution-mode initialization followed by one complete operation."""

        if self.external_reference is not None:
            start_time = time.perf_counter()
            assert isinstance(self.runner, GdalRunner)
            self.runner._execute()
            return time.perf_counter() - start_time

        self.runner.close()
        assert self.execution_mode is not None
        fresh_runner = BenchmarkRunner(self.execution_mode, self.config)
        start_time = time.perf_counter()
        try:
            fresh_runner.start()
            fresh_runner._execute(self.operation)
        finally:
            elapsed_time_s = time.perf_counter() - start_time
            fresh_runner.close()
        self.runner = fresh_runner
        return elapsed_time_s

    def track_peak_process_tree_mem_mb(self, parameter: int) -> float:
        """Measure peak memory for the benchmark process and execution-mode children."""

        if isinstance(self.runner, GdalRunner):
            return self.runner.run().peak_process_tree_mem_mb
        return self.runner.run(self.operation).peak_process_tree_mem_mb


# ASV reads tracker units from method attributes when labelling stored values
setattr(_ComparisonBenchmark.track_end_to_end_time_s, "unit", "seconds")
setattr(_ComparisonBenchmark.track_peak_process_tree_mem_mb, "unit", "MB")


class _InterpolationPointCount(_ComparisonBenchmark):
    """Keep raster and chunk sizes fixed while varying interpolated points."""

    param_names = ["interpolated_points"]
    params = [asv_parameter_values([256, 2048, 16384], pr_check_value=256)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected point count in an otherwise fixed configuration."""

        return BenchmarkConfig(shape=(2048, 2048), chunks=(512, 512), ninterp=parameter)


class _ReprojectionRasterSize(_ComparisonBenchmark):
    """Keep chunk size fixed while varying input and output raster size."""

    param_names = ["raster_size"]
    params = [asv_parameter_values([1024, 2048, 4096], pr_check_value=1024)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size in an otherwise fixed configuration."""

        return BenchmarkConfig(shape=(parameter, parameter), chunks=(512, 512))


class _FilterChunkSize(_ComparisonBenchmark):
    """Keep raster size and filter window fixed while varying square chunks."""

    param_names = ["chunk_size"]
    params = [asv_parameter_values([256, 512, 1024], pr_check_value=1024)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected chunk size in an otherwise fixed configuration."""

        return BenchmarkConfig(shape=(2048, 2048), chunks=(parameter, parameter))


class _PolygonizationRasterSize(_ComparisonBenchmark):
    """Keep connected-region count fixed while varying raster size."""

    param_names = ["raster_size"]
    params = [asv_parameter_values([1024, 2048, 4096], pr_check_value=1024)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size around a fixed set of regions."""

        return BenchmarkConfig(
            shape=(parameter, parameter),
            chunks=(512, 512),
            polygon_regions_per_axis=21,
        )


class _RasterizationRasterSize(_ComparisonBenchmark):
    """Keep vector complexity fixed while varying output raster size."""

    param_names = ["raster_size"]
    params = [asv_parameter_values([1024, 2048, 4096], pr_check_value=1024)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size around a fixed vector input."""

        return BenchmarkConfig(
            shape=(parameter, parameter),
            chunks=(512, 512),
            vector_features_per_axis=51,
        )


class _SubsampleSize(_ComparisonBenchmark):
    """Keep raster and chunks fixed while varying the selected value count."""

    param_names = ["subsample_size"]
    params = [asv_parameter_values([256, 2048, 16384], pr_check_value=256)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected value count in an otherwise fixed configuration."""

        return BenchmarkConfig(shape=(2048, 2048), chunks=(512, 512), subsample_size=parameter)


class _GriddingRasterSize(_ComparisonBenchmark):
    """Keep one gridding method and point count fixed while varying raster size."""

    param_names = ["raster_size"]
    params = [asv_parameter_values([512, 1024, 2048], pr_check_value=512)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size around the common source point input."""

        # Keep the point count fixed so only the method and its required support distance differ
        method_distances = {
            "nearest": float("inf"),
            "linear": float("inf"),
            "idw": 16.0,
            "mean": 16.0,
        }
        if self.operation_method not in method_distances:
            raise ValueError(f"No gridding fixture is defined for method {self.operation_method!r}")
        return BenchmarkConfig(
            shape=(parameter, parameter),
            chunks=(512, 512),
            point_features_per_axis=_GRID_POINTS_PER_AXIS[self.operation_method],
            grid_dist_nodata_pixel=method_distances[self.operation_method],
        )


class _NearestGriddingPointCount(_ComparisonBenchmark):
    """Keep raster size fixed while varying source points for nearest gridding."""

    param_names = ["points_per_axis"]
    params = [asv_parameter_values([3, 9, 33], pr_check_value=3)]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected point count in an otherwise fixed configuration."""

        return BenchmarkConfig(
            shape=(1024, 1024),
            chunks=(512, 512),
            point_features_per_axis=parameter,
            grid_dist_nodata_pixel=float("inf"),
        )


class _NumbaWorkerIntegration(_GriddingRasterSize):
    """Exercise each Numba kernel once in Dask and multiprocessing workers."""

    params = [asv_parameter_values([1024], pr_check_value=512)]


# Select the input axis and fixture configuration used by each named comparison group
_SCENARIO_BASES: dict[str, type[_ComparisonBenchmark]] = {
    "interpolation-point-count": _InterpolationPointCount,
    "reprojection-raster-size": _ReprojectionRasterSize,
    "filter-chunk-size": _FilterChunkSize,
    "polygonization-raster-size": _PolygonizationRasterSize,
    "rasterization-raster-size": _RasterizationRasterSize,
    "subsample-size": _SubsampleSize,
    "gridding-raster-size": _GriddingRasterSize,
    "gridding-point-count": _NearestGriddingPointCount,
    "worker-integration": _NumbaWorkerIntegration,
}


def _register_asv_classes() -> None:
    """Create stable public ASV classes from the deduplicated case registry."""

    for case in (*BENCHMARK_CASES, *EXTERNAL_REFERENCE_CASES):
        class_name = case.benchmark_class
        if class_name in globals():
            raise ValueError(f"Duplicate generated ASV benchmark class: {class_name}")
        base = _SCENARIO_BASES[case.comparison_group]
        globals()[class_name] = type(
            class_name,
            (base,),
            {
                "__module__": __name__,
                "__doc__": f"Measure the registered {case.operation} benchmark case.",
            },
        )


# ASV discovers public module classes, so create one class for every registered case after defining the bases
_register_asv_classes()
