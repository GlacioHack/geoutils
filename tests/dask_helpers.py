"""Helper functions for tests involving a Dask client."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import pytest

Result = TypeVar("Result")


def run_with_dask_client(
    function: Callable[[Any], Result], *, client_options: dict[str, Any] | None = None, **cluster_options: Any
) -> Result:
    """Run a calculation with a local Dask client and await shutdown before returning."""

    pytest.importorskip("distributed")
    from distributed import Client, LocalCluster
    from distributed.compatibility import asyncio_run
    from distributed.config import get_loop_factory
    from distributed.scheduler import WorkerStatusPlugin
    from tornado.ioloop import IOLoop

    tornado_loop: IOLoop | None = None

    async def run_cluster() -> Result:
        """Run the calculation and await client and cluster shutdown on their event loop."""

        nonlocal tornado_loop
        tornado_loop = IOLoop.current()
        cluster = LocalCluster(asynchronous=True, **cluster_options)
        client = None
        try:
            async with cluster:
                async with Client(cluster, asynchronous=True, **(client_options or {})) as client:
                    return await asyncio.to_thread(function, client)
        finally:
            # Dask can close its client connection before the message sender has stopped
            # test_close_closed() and test_close_twice() cover this extra close:
            # https://github.com/dask/distributed/blob/2026.8.0/distributed/tests/test_batched.py#L125-L155
            # close() wakes the sender and waits for it to stop:
            # https://github.com/dask/distributed/blob/2026.8.0/distributed/batched.py#L164-L175
            if client is not None and client.scheduler_comm is not None:
                await client.scheduler_comm.close()

            # We need to stop the worker status sender too, while the event loop can still run its cleanup
            if cluster.scheduler is not None:
                for plugin in tuple(cluster.scheduler.plugins.values()):
                    if isinstance(plugin, WorkerStatusPlugin):
                        await plugin.bcomm.close()

    # Let Dask choose a compatible event loop, including for Windows (avoid CI issues)
    try:
        return asyncio_run(run_cluster(), loop_factory=get_loop_factory())
    finally:
        # We need to close Tornado too, especially on Windows where it can leave a background thread running
        # Follow Dask's _run_and_close_tornado() test helper:
        # https://github.com/dask/distributed/blob/2026.8.0/distributed/utils_test.py#L371-L382
        if tornado_loop is not None:
            tornado_loop.close(all_fds=True)
