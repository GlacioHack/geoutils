# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""List the operations and backends shared by ASV and the large data tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BackendName = Literal["dask", "multiprocessing"]
OperationName = Literal[
    "crop",
    "translate",
    "copy",
    "filter",
    "reproject",
    "statistics",
    "subsample",
    "interp_points",
    "polygonize",
    "write",
    "rasterize",
    "create_mask",
    "grid",
]


@dataclass(frozen=True)
class OperationCase:
    """Describe one public operation and the out-of-core backends it supports."""

    operation: OperationName
    backends: tuple[BackendName, ...]
    expected_value: float


# Keep public out-of-core claims in one executable list
OPERATION_CASES: tuple[OperationCase, ...] = (
    OperationCase("crop", ("dask",), 1),
    OperationCase("translate", ("dask",), 1),
    OperationCase("copy", ("dask",), 1),
    OperationCase("filter", ("dask", "multiprocessing"), 1),
    OperationCase("reproject", ("dask", "multiprocessing"), 1),
    OperationCase("statistics", ("dask",), 1),
    OperationCase("subsample", ("dask", "multiprocessing"), 1),
    OperationCase("interp_points", ("dask", "multiprocessing"), 1),
    OperationCase("polygonize", ("dask", "multiprocessing"), 1),
    OperationCase("write", ("dask",), 1),
    OperationCase("rasterize", ("dask", "multiprocessing"), 1),
    OperationCase("create_mask", ("dask", "multiprocessing"), 1),
    OperationCase("grid", ("dask", "multiprocessing"), 1),
)

# Flatten supported pairs for ASV and pytest without creating invalid combinations
OPERATION_BY_NAME = {case.operation: case for case in OPERATION_CASES}
OPERATION_BENCHMARK_CASES = tuple(
    f"{backend}-{case.operation}" for case in OPERATION_CASES for backend in case.backends
)


def split_operation_case(case_name: str) -> tuple[BackendName, OperationName]:
    """Split one stable benchmark identifier into backend and operation names."""

    # A single case parameter avoids the invalid product of all backends and operations
    backend, operation = case_name.split("-", maxsplit=1)
    if backend not in ("dask", "multiprocessing") or operation not in OPERATION_BY_NAME:
        raise ValueError(f"Unknown benchmark operation case: {case_name}")
    return backend, operation  # type: ignore[return-value]
