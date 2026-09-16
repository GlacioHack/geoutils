"""Functions to test the clusters."""

import multiprocessing
import time
from typing import Any

import pytest

from geoutils.multiproc.cluster import (
    AbstractCluster,
    BasicCluster,
    ClusterGenerator,
    MpCluster,
    _map_bounded,
)


# Sample function for testing
def sample_function(x: float, y: float) -> float:
    return x + y


# Function to simulate a long task
def long_running_task(x: float) -> float:
    time.sleep(0.01)
    return x * 2


def wait_for_release(release_event: Any, value: int) -> int:
    """Wait for the parent process to release this worker task, then return a value."""

    if not release_event.wait(timeout=30):
        raise TimeoutError("The parent process did not release the worker task.")
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

    def test_mp_cluster_completion_order(self) -> None:
        """Checks that iter_completed() yields a finished task before an earlier blocked task."""

        # Keep the first task blocked so process startup and scheduling cannot let it finish before the second task
        with multiprocessing.Manager() as manager:
            release_first = manager.Event()
            with ClusterGenerator("multiprocessing", nb_workers=2) as cluster:
                assert isinstance(cluster, MpCluster)

                # Submit the blocked task first and check that the immediately returning second task is yielded first
                futures = [
                    cluster.submit(wait_for_release, release_first, 0),
                    cluster.submit(sample_function, 0, 1),
                ]
                completed = cluster.iter_completed(futures)
                first_completed = next(completed)

                # Release the first task only after observing the second task, then collect the remaining result
                release_first.set()
                remaining_completed = list(completed)

        # The runnable second task must be reported before the blocked first task
        assert [first_completed, *remaining_completed] == [(1, 1), (0, 0)]

    def test_map_bounded__refills_workers_and_keeps_input_order(self) -> None:
        """Checks that bounded mapping submits replacement work early and returns results in input order."""

        class ReverseCompletionCluster(AbstractCluster):
            """Return the newest pending result first while recording submitted inputs."""

            def __init__(self) -> None:
                super().__init__()
                self.submitted: list[int] = []

            def submit(self, fun: Any, *args: Any, **kwargs: Any) -> Any:
                """Calculate and record one immediate result."""

                self.submitted.append(args[0])
                return fun(*args, **kwargs)

            def iter_completed(self, futures: list[Any]) -> Any:
                """Return the most recently submitted pending result."""

                yield len(futures) - 1, futures[-1]

            def close(self) -> None:
                """Close this in-memory test cluster."""

        # Delay the first returned result by reporting the second pending call as completed first
        cluster = ReverseCompletionCluster()
        mapped = _map_bounded(cluster, lambda value: value * 2, ((value,) for value in range(6)), max_pending=2)
        first = next(mapped)

        # Replacement calls start before the first ordered result is returned, while results still follow input order
        assert first == (0, 0)
        assert cluster.submitted == [0, 1, 2, 3]
        assert [first, *mapped] == [(index, index * 2) for index in range(6)]

    def test_mp_cluster_termination(self) -> None:
        # Test that the pool terminates correctly after closing
        cluster = ClusterGenerator("multiprocessing", nb_workers=2)
        assert isinstance(cluster, MpCluster)

        # Close the cluster
        cluster.close()

        # Expect an error when trying to launch a task after closing
        with pytest.raises(ValueError):
            cluster.submit(sample_function, 2, 3)
