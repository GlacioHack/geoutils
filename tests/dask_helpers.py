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
        if tornado_loop is not None:
            tornado_loop.close(all_fds=True)
