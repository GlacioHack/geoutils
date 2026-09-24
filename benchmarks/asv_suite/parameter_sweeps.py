"""Generate ASV classes from operation-local one-dimensional parameter sweeps."""

from __future__ import annotations

import tempfile
import time
from typing import cast

from benchmarks.asv_suite import asv_pr_check_enabled
from benchmarks.gdal_comparison.commands import ComparisonOperation
from benchmarks.gdal_comparison.runner import GdalRunner
from benchmarks.pdal_comparison.commands import PdalComparisonOperation
from benchmarks.pdal_comparison.runner import PdalRunner
from benchmarks.workflows.config import (
    BenchmarkCase,
    BenchmarkConfig,
    CalculationEngine,
    ExecutionMode,
    ExternalReference,
    ExternalReferenceCase,
    OperationName,
    OperationStrategyName,
    Parameter,
    Sweep,
)
from benchmarks.workflows.operations import (
    SWEEPS,
    format_api_label,
)
from benchmarks.workflows.runner import BenchmarkRunner

#####################################
# ASV measurements and input sizes
#####################################


# Numeric input ranges and configuration changes live in operation modules; this class keeps measurements shared
class _ComparisonBenchmark:
    """Share ASV settings and complete result computation across operation sweeps."""

    timeout = 900
    number = 1
    repeat = 2
    rounds = 1
    warmup_time = 0
    sweep: Sweep
    case: BenchmarkCase | ExternalReferenceCase
    operation: OperationName
    operation_method: str | None
    calculation_engine: CalculationEngine | None
    operation_strategy: OperationStrategyName | None
    execution_mode: ExecutionMode | None
    external_reference: ExternalReference | None

    def make_config(self, parameter: Parameter) -> BenchmarkConfig:
        """Build one configuration from the operation-local sweep."""

        return self.sweep.make_config(parameter, self.case, asv_pr_check_enabled())

    def setup(self, parameter: Parameter) -> None:
        """Prepare deterministic files and initialize one execution case."""

        selected_case = self.case
        case = selected_case if isinstance(selected_case, BenchmarkCase) else None
        reference_case = selected_case if isinstance(selected_case, ExternalReferenceCase) else None
        if asv_pr_check_enabled() and not selected_case.pr_check:
            raise NotImplementedError("Benchmark case omitted from the pull-request sample")

        # Strategies only identify how Dask or multiprocessing coordinates chunks
        self.operation = selected_case.operation
        self.operation_method = selected_case.method
        self.operation_strategy = selected_case.strategy
        self.calculation_engine = case.engine if case is not None else None
        self.execution_mode = case.execution if case is not None else None
        self.external_reference = reference_case.external_reference if reference_case is not None else None

        # Input generation remains outside all three measured boundaries
        self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-comparison-")
        self.config = self.make_config(parameter)
        self.config.point_output_driver = selected_case.output_driver
        self.config.operation_method = self.operation_method
        self.config.calculation_engine = self.calculation_engine
        self.config.operation_strategy = self.operation_strategy
        self.config.directory = self._tmpdir.name
        self.sources = BenchmarkRunner("eager", self.config).prepare_sources()

        if self.external_reference == "gdal_cli":
            gdal_operation = cast(ComparisonOperation, self.operation)
            self.runner: BenchmarkRunner | GdalRunner | PdalRunner = GdalRunner(
                gdal_operation, self.config, self.sources
            )
        elif self.external_reference == "pdal_cli":
            pdal_operation = cast(PdalComparisonOperation, self.operation)
            self.runner = PdalRunner(pdal_operation, self.config, self.sources)
        else:
            assert self.execution_mode is not None
            self.runner = BenchmarkRunner(self.execution_mode, self.config).start()

    def teardown(self, parameter: Parameter) -> None:
        """Stop workers and remove generated source, output and spill files."""

        if not hasattr(self, "runner"):
            return
        self.runner.close()
        if self.sources is not self.runner:
            self.sources.close()
        self._tmpdir.cleanup()

    def time_operation(self, parameter: Parameter) -> None:
        """Measure a complete operation after execution-mode initialization."""

        if isinstance(self.runner, (GdalRunner, PdalRunner)):
            self.runner._execute()
        else:
            self.runner._execute(self.operation)

    def track_end_to_end_time_s(self, parameter: Parameter) -> float:
        """Measure execution-mode initialization followed by one complete operation."""

        if self.external_reference is not None:
            start_time = time.perf_counter()
            assert isinstance(self.runner, (GdalRunner, PdalRunner))
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

    def track_process_tree_mem_increase_mb(self, parameter: Parameter) -> float:
        """Measure peak memory increase above the initialized process-tree baseline."""

        if isinstance(self.runner, (GdalRunner, PdalRunner)):
            return self.runner.run().process_tree_mem_increase_mb
        return self.runner.run(self.operation).process_tree_mem_increase_mb


# ASV reads tracker units from method attributes when labelling stored values
setattr(_ComparisonBenchmark.track_end_to_end_time_s, "unit", "seconds")
setattr(_ComparisonBenchmark.track_process_tree_mem_increase_mb, "unit", "MB")


#####################################
# Public ASV class registration
#####################################


def _register_asv_classes() -> None:
    """Create stable public ASV classes from discovered cases and sweeps."""

    for sweep in SWEEPS:
        base = sweep.harness or _ComparisonBenchmark
        for case in (*sweep.cases, *sweep.references):
            class_name = case.benchmark_class
            if class_name in globals():
                raise ValueError(f"Duplicate generated ASV benchmark class: {class_name}")
            attributes = {
                "__module__": __name__,
                "__doc__": f"Measure the registered {case.operation} benchmark case.",
                "sweep": sweep,
                "case": case,
                "param_names": [sweep.param_name],
                "params": [list(sweep.parameters(asv_pr_check_enabled()))],
            }
            if isinstance(case, BenchmarkCase):
                attributes["pretty_name"] = format_api_label(case.operation, case)
            globals()[class_name] = type(class_name, (base,), attributes)


# ASV discovers public module classes, so create one class for every registered case after defining the bases
_register_asv_classes()
