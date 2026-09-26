"""Generate the necessary classes for ASV to detect and measures all operations during benchmarks."""

from __future__ import annotations

import gc
import tempfile
import time
from functools import wraps
from typing import Any

from benchmarks.asv_suite import asv_parameter_values
from benchmarks.workflows.config import Parameter, with_directory
from benchmarks.workflows.core import Benchmark, Case
from benchmarks.workflows.operations import BENCHMARKS, format_api_label
from benchmarks.workflows.runner import BenchmarkRunner


class _ASVBenchmarkBase:
    """This class defines ASV settings + methods, used across all operation benchmarks."""

    # Allow one complete operation to run for up to 15 minutes
    timeout = 900

    # Run once per timing sample, collect two samples and avoid an extra warm-up pass
    number = 1
    repeat = 2
    rounds = 1
    warmup_time = 0
    benchmark: Benchmark
    case: Case

    # ASV passes one value to these methods for parameterized benchmarks and uses the default for fixed benchmarks
    def setup(self, parameter: Parameter | None = None) -> None:
        """Prepare inputs and initialize execution case."""

        # Input generation remains outside all three measured boundaries
        self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-benchmark-")
        config = self.benchmark.make_config(parameter, self.case)
        self.config = with_directory(config, self._tmpdir.name)

        # Source creation and worker startup do not belong to operation measurements
        self.runner = BenchmarkRunner(self.benchmark.operation, self.case, self.config).start()

    def teardown(self, parameter: Parameter | None = None) -> None:
        """Stop workers and remove generated source, output and spill files."""

        if not hasattr(self, "runner"):
            return

        # ASV invokes teardown independently after the time and memory benchmarks
        self.runner.close()
        self._tmpdir.cleanup()

        # Collect closed Dask event-loop cycles before Python tears down the modules used by their tracebacks
        if self.runner.backend == "dask":
            gc.collect()

    def time_operation(self, parameter: Parameter | None = None) -> None:
        """Measure a complete operation after initialization."""

        # The time_ prefix tells ASV to time this method automatically
        # Every large output is written before the measured method returns
        self.runner._execute()

    def track_end_to_end_time_s(self, parameter: Parameter | None = None) -> float:
        """Measure initialization followed by one complete operation."""

        # The track_ prefix tells ASV to record the returned numeric measurement
        self.runner.close()
        fresh_runner = BenchmarkRunner(self.benchmark.operation, self.case, self.config)
        start_time = time.perf_counter()
        try:
            fresh_runner.start()
            fresh_runner._execute()
        finally:
            elapsed_time_s = time.perf_counter() - start_time
            fresh_runner.close()
        self.runner = fresh_runner
        return elapsed_time_s

    def track_process_tree_mem_increase_mb(self, parameter: Parameter | None = None) -> float:
        """Measure peak memory increase above the initialized process-tree baseline."""

        # The track_ prefix tells ASV to record the returned numeric measurement
        # Profiling repeats the same complete operation with process-tree sampling enabled
        return self.runner.run().process_tree_mem_increase_mb


# Label time in seconds and memory increase in MBs
setattr(_ASVBenchmarkBase.track_end_to_end_time_s, "unit", "seconds")
setattr(_ASVBenchmarkBase.track_process_tree_mem_increase_mb, "unit", "MB")


#####################################
# Public ASV class registration
#####################################


def _named_method(method: Any, benchmark_name: str) -> Any:
    """Copy one generated measurement method with its stable ASV identifier."""

    @wraps(method)
    def measured(self: Any, *parameters: Any) -> Any:
        return method(self, *parameters)

    setattr(measured, "benchmark_name", benchmark_name)
    return measured


def _register_asv_classes() -> None:
    """Create ASV classes from benchmarks and cases that were discovered in workflows/operations/."""

    for benchmark in BENCHMARKS:
        # One generated class per case avoids the invalid product of all execution modes and operations
        for case in benchmark.cases:
            class_name = benchmark.benchmark_class(case)
            if class_name in globals():
                raise ValueError(f"Duplicate generated ASV benchmark class: {class_name}")
            attributes: dict[str, Any] = {
                "__module__": __name__,
                "__doc__": f"Measure the registered {benchmark.operation.name} benchmark case.",
                "benchmark": benchmark,
                "case": case,
            }
            if benchmark.parameter_name is not None:
                attributes["param_names"] = [benchmark.parameter_name]
                attributes["params"] = [asv_parameter_values(benchmark.values)]
                for method_name in (
                    "time_operation",
                    "track_end_to_end_time_s",
                    "track_process_tree_mem_increase_mb",
                ):
                    method = getattr(_ASVBenchmarkBase, method_name)
                    stable_name = f"asv_suite.parameter_sweeps.{class_name}.{method_name}"
                    attributes[method_name] = _named_method(method, stable_name)
            if case.implementation == "geoutils":
                attributes["pretty_name"] = format_api_label(benchmark.operation, case)
            globals()[class_name] = type(class_name, (_ASVBenchmarkBase,), attributes)


# ASV discovers public module classes, so create one class for every registered case after defining the bases
_register_asv_classes()
