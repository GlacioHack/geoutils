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

    async def run_cluster() -> Result:
        async with LocalCluster(asynchronous=True, **cluster_options) as cluster:
            async with Client(cluster, asynchronous=True, **(client_options or {})) as client:
                return await asyncio.to_thread(function, client)

    # Let Dask choose a compatible event loop, including for Windows (avoid CI issues)
    return asyncio_run(run_cluster(), loop_factory=get_loop_factory())
