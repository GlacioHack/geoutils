"""Functions to test the clusters."""

import multiprocessing
import time
from typing import Any

import pytest

from geoutils.multiproc.cluster import (
    BasicCluster,
    ClusterGenerator,
    MpCluster,
)


# Sample function for testing
def sample_function(x: float, y: float) -> float:
    return x + y


# Function to simulate a long task
def long_running_task(x: float) -> float:
    time.sleep(0.01)
    return x * 2


def delayed_value(start_barrier: Any, delay: float, value: int) -> int:
    """Wait for both worker tasks to start, then return a value after a controlled delay."""

    start_barrier.wait(timeout=30)
    time.sleep(delay)
    return value


class TestClusterGenerator:
    """Test module for synchronous and process-based clusters through their shared interface."""

    def test_basic_cluster(self) -> None:
        # Test that tasks are run synchronously in BasicCluster
        cluster = ClusterGenerator(name="basic")
        assert isinstance(cluster, BasicCluster)

        result = cluster.submit(sample_function, 2, 3)
        assert result == 5

    def test_mp_cluster_task(self) -> None:
        # Test that tasks are launched asynchronously in MpCluster
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        future = cluster.submit(sample_function, 2, 3)
        result = cluster.compute(future)
        assert result == 5

    def test_mp_cluster_parallelism(self) -> None:
        # Test that multiple tasks are run in parallel
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        futures = [cluster.submit(long_running_task, i) for i in range(4)]
        results = cluster.gather(futures)
        assert results == [0, 2, 4, 6]

    def test_mp_cluster__default_start_method(self) -> None:
        """Checks that MpCluster uses "forkserver" when available, otherwise falls back to spawn."""

        # Select the platform default
        available_methods = multiprocessing.get_all_start_methods()
        expected_method = "forkserver" if "forkserver" in available_methods else "spawn"

        # Start one worker and check that the selected context can execute an ordinary task
        with MpCluster({"nb_workers": 1}) as cluster:
            assert cluster.start_method == expected_method
            assert cluster.compute(cluster.submit(sample_function, 2, 3)) == 5

    def test_mp_cluster__explicit_start_method(self) -> None:
        """Checks that MpCluster accepts an available start method selected by the user."""

        # Spawn is available on every supported platform and provides a clean worker process
        with MpCluster({"nb_workers": 1}, start_method="spawn") as cluster:
            assert cluster.start_method == "spawn"
            assert cluster.compute(cluster.submit(sample_function, 2, 3)) == 5

    def test_mp_cluster__error_unavailable_start_method(self) -> None:
        """Checks that MpCluster rejects a start method unavailable on the current platform."""

        # Reject the value before creating a worker pool and list the methods the runtime supports
        with pytest.raises(ValueError, match="is not available on this platform"):
            MpCluster({"nb_workers": 1}, start_method="unavailable")  # type: ignore[arg-type]

    def test_mp_cluster_completion_order(self) -> None:
        """Checks that iter_completed() yields a later submitted task when it finishes first."""

        # Start both tasks before either delay begins. Windows starts fresh worker processes, and one worker can
        # otherwise run a complete task before the other worker has finished starting
        with multiprocessing.Manager() as manager:
            start_barrier = manager.Barrier(2)
            with ClusterGenerator("multiprocessing", nb_workers=2) as cluster:
                assert isinstance(cluster, MpCluster)

                # Submit the longer task first, then collect results as each task finishes
                futures = [
                    cluster.submit(delayed_value, start_barrier, 0.2, 0),
                    cluster.submit(delayed_value, start_barrier, 0.01, 1),
                ]
                completed = list(cluster.iter_completed(futures))

        # The short second task must be reported before the long first task
        assert completed == [(1, 1), (0, 0)]

    def test_mp_cluster_termination(self) -> None:
        # Test that the pool terminates correctly after closing
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        # Close the cluster
        cluster.close()

        # Expect an error when trying to launch a task after closing
        with pytest.raises(ValueError):
            cluster.submit(sample_function, 2, 3)
