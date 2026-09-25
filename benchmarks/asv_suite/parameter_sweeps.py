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
    case: BenchmarkCase

    def make_config(self, parameter: Parameter) -> BenchmarkConfig:
        """Build one configuration from the operation-local sweep."""

        return self.sweep.make_config(parameter, self.case, asv_pr_check_enabled())

    def setup(self, parameter: Parameter) -> None:
        """Prepare deterministic files and initialize one execution case."""

        selected_case = self.case
        if asv_pr_check_enabled() and not selected_case.pr_check:
            raise NotImplementedError("Benchmark case omitted from the pull-request sample")

        # Input generation remains outside all three measured boundaries
        self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-comparison-")
        self.config = self.make_config(parameter)
        self.config.directory = self._tmpdir.name
        self.sources = BenchmarkRunner("inmem", self.config).prepare_sources()

        if selected_case.external_reference == "gdal_cli":
            gdal_operation = cast(ComparisonOperation, selected_case.operation)
            self.runner: BenchmarkRunner | GdalRunner | PdalRunner = GdalRunner(
                gdal_operation, self.config, self.sources
            )
        elif selected_case.external_reference == "pdal_cli":
            pdal_operation = cast(PdalComparisonOperation, selected_case.operation)
            self.runner = PdalRunner(pdal_operation, self.config, self.sources)
        else:
            assert selected_case.execution is not None
            self.runner = BenchmarkRunner(selected_case.execution, self.config).start()

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
            self.runner._execute(self.case.operation)

    def track_end_to_end_time_s(self, parameter: Parameter) -> float:
        """Measure execution-mode initialization followed by one complete operation."""

        if self.case.external_reference is not None:
            start_time = time.perf_counter()
            assert isinstance(self.runner, (GdalRunner, PdalRunner))
            self.runner._execute()
            return time.perf_counter() - start_time

        self.runner.close()
        assert self.case.execution is not None
        fresh_runner = BenchmarkRunner(self.case.execution, self.config)
        start_time = time.perf_counter()
        try:
            fresh_runner.start()
            fresh_runner._execute(self.case.operation)
        finally:
            elapsed_time_s = time.perf_counter() - start_time
            fresh_runner.close()
        self.runner = fresh_runner
        return elapsed_time_s

    def track_process_tree_mem_increase_mb(self, parameter: Parameter) -> float:
        """Measure peak memory increase above the initialized process-tree baseline."""

        if isinstance(self.runner, (GdalRunner, PdalRunner)):
            return self.runner.run().process_tree_mem_increase_mb
        return self.runner.run(self.case.operation).process_tree_mem_increase_mb


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
            class_name = sweep.benchmark_class(case)
            if class_name in globals():
                raise ValueError(f"Duplicate generated ASV benchmark class: {class_name}")
            attributes = {
                "__module__": __name__,
                "__doc__": f"Measure the registered {case.operation} benchmark case.",
                "sweep": sweep,
                "case": case,
                "param_names": [sweep.axis.name],
                "params": [list(sweep.axis.parameters(asv_pr_check_enabled()))],
            }
            if case.external_reference is None:
                attributes["pretty_name"] = format_api_label(case)
            globals()[class_name] = type(class_name, (base,), attributes)


# ASV discovers public module classes, so create one class for every registered case after defining the bases
_register_asv_classes()
