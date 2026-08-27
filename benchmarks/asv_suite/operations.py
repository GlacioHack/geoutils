"""Measure repeatable time and RAM for every registered operation and backend."""

from __future__ import annotations

from benchmarks.workflows.registry import (
    OPERATION_BENCHMARK_CASES,
    split_operation_case,
)
from benchmarks.workflows.runner import BenchmarkConfig, BenchmarkRunner


class OperationBenchmarks:
    """Measure every advertised backend operation at one fixed configuration."""

    # Allow one complete operation to run for up to 15 minutes
    timeout = 900

    # Run once per timing sample, collect three samples and avoid an extra warm-up pass
    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0

    # ASV passes every registry identifier to setup and both measurement methods
    param_names = ["case"]
    params = [OPERATION_BENCHMARK_CASES]

    def setup(self, case: str) -> None:
        """Prepare deterministic files and one backend outside the measured region."""

        # One case identifier avoids the invalid product of all backends and operations
        backend, self.operation = split_operation_case(case)
        self.runner = BenchmarkRunner(
            backend,
            BenchmarkConfig(
                shape=(2048, 2048),
                chunks=(512, 512),
                memory_limit="1GB",
                subsample_size=2048,
                ninterp=2048,
                profile_interval=0.05,
            ),
        )

        # Source creation and worker startup do not belong to operation measurements
        self.runner.start()

    def teardown(self, case: str) -> None:
        """Stop workers and remove generated source, output and spill files."""

        # ASV invokes teardown independently after the time and memory benchmarks
        self.runner.close()

    def time_operation(self, case: str) -> None:
        """Measure elapsed time through the operation's final output computation."""

        # The time_ prefix tells ASV to time this method automatically
        # Every large output is written before the measured method returns
        self.runner._execute(self.operation)

    def track_peak_process_tree_rss_mb(self, case: str) -> float:
        """Measure aggregate peak RAM for the client and all backend processes."""

        # The track_ prefix tells ASV to record the returned numeric measurement
        # Profiling repeats the same complete operation with process-tree sampling enabled
        result = self.runner.run(self.operation)
        return result.peak_process_tree_rss_mb


# ASV reads this unit attribute when labelling the stored tracker values
setattr(OperationBenchmarks.track_peak_process_tree_rss_mb, "unit", "MB")
