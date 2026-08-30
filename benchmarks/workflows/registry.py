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

"""List supported benchmark dimensions and resolve one valid operation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Define the dimensions supported by GeoUtils and tested in benchmarks: execution modes,
# calculation engines, operation names, operation methods and chunk strategies
ExecutionMode = Literal["eager", "dask", "multiprocessing"]
CalculationEngine = Literal["scipy", "numba", "rasterio"]
OperationStrategyName = Literal["sequential", "topk", "label_union", "label_stitch", "geometry_stitch"]
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


# Store which methods and engines each operation supports, which chunk strategies it offers,
# and which execution modes are tested
@dataclass(frozen=True)
class OperationMethod:
    """Describe one benchmarked method and its supported calculation engines."""

    operation: OperationName
    method: str | None
    calculation_engines: tuple[CalculationEngine, ...]
    default: bool = False


@dataclass(frozen=True)
class OperationStrategy:
    """Describe one approach for coordinating or reconciling chunked work."""

    operation: OperationName
    strategy: OperationStrategyName
    default: bool = False


@dataclass(frozen=True)
class OperationCase:
    """Describe one public operation and the out-of-core execution modes it supports."""

    operation: OperationName
    execution_modes: tuple[ExecutionMode, ...]
    expected_value: float


# List the supported method and calculation-engine combinations for each numerical operation
# Single-method operations stay explicit so the engine is always recorded in benchmark results
OPERATION_METHODS: tuple[OperationMethod, ...] = (
    OperationMethod("interp_points", "linear", ("scipy",), default=True),
    OperationMethod("reproject", "nearest", ("rasterio",), default=True),
    OperationMethod("filter", "mean", ("scipy",), default=True),
    OperationMethod("polygonize", None, ("rasterio",), default=True),
    OperationMethod("rasterize", None, ("rasterio",), default=True),
    OperationMethod("grid", "nearest", ("scipy", "numba"), default=True),
    OperationMethod("grid", "linear", ("scipy",)),
    OperationMethod("grid", "idw", ("scipy", "numba")),
    OperationMethod("grid", "mean", ("scipy", "numba")),
)

# List the alternative ways chunked operations select or reconcile results; eager execution has no strategy
OPERATION_STRATEGIES: tuple[OperationStrategy, ...] = (
    OperationStrategy("subsample", "sequential"),
    OperationStrategy("subsample", "topk", default=True),
    OperationStrategy("polygonize", "label_union"),
    OperationStrategy("polygonize", "label_stitch", default=True),
    OperationStrategy("polygonize", "geometry_stitch"),
)


# List each operation tested out of core, its supported execution modes and its expected result value
# Both the fixed ASV benchmarks and large-data tests use this coverage list
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

# Map an operation name to its expected result and supported execution modes
OPERATION_BY_NAME = {case.operation: case for case in OPERATION_CASES}

# Build identifiers such as "dask-grid" only for execution mode and operation pairs that are supported
OPERATION_BENCHMARK_CASES = tuple(
    f"{execution_mode}-{case.operation}" for case in OPERATION_CASES for execution_mode in case.execution_modes
)


def resolve_operation_parameters(
    operation: OperationName,
    method: str | None = None,
    calculation_engine: CalculationEngine | None = None,
    strategy: OperationStrategyName | None = None,
    execution_mode: ExecutionMode | None = None,
) -> tuple[str | None, CalculationEngine | None, OperationStrategyName | None]:
    """Resolve and validate the method, engine and strategy for one operation."""

    # Find the requested method, or the registered default, before checking that its engine is supported
    operation_methods = tuple(
        specification for specification in OPERATION_METHODS if specification.operation == operation
    )
    if not operation_methods:
        if method is not None or calculation_engine is not None:
            raise ValueError(f"Operation {operation!r} has no registered benchmark method or calculation engine")
    else:
        if method is None:
            selected = tuple(specification for specification in operation_methods if specification.default)
        else:
            selected = tuple(specification for specification in operation_methods if specification.method == method)
        if len(selected) != 1:
            raise ValueError(f"Expected one benchmark method for {operation!r}/{method!r}")

        specification = selected[0]
        method = specification.method
        calculation_engine = calculation_engine or specification.calculation_engines[0]
        if calculation_engine not in specification.calculation_engines:
            raise ValueError(
                f"Engine {calculation_engine!r} does not support benchmark method {operation!r}/{method!r}"
            )

    # Chunk strategies are selected independently and rejected for eager execution
    operation_strategies = tuple(
        specification for specification in OPERATION_STRATEGIES if specification.operation == operation
    )
    if not operation_strategies:
        if strategy is not None:
            raise ValueError(f"Operation {operation!r} has no registered benchmark strategy")
    elif execution_mode in (None, "eager"):
        if strategy is not None:
            raise ValueError(f"Strategy {strategy!r} only applies to chunked execution of {operation!r}")
        strategy = None
    else:
        if strategy is None:
            selected_strategies = tuple(
                specification for specification in operation_strategies if specification.default
            )
        else:
            selected_strategies = tuple(
                specification for specification in operation_strategies if specification.strategy == strategy
            )
        if len(selected_strategies) != 1:
            raise ValueError(f"Expected one benchmark strategy for {operation!r}/{strategy!r}")
        strategy = selected_strategies[0].strategy

    return method, calculation_engine, strategy


def split_operation_case(case_name: str) -> tuple[ExecutionMode, OperationName]:
    """Split one stable benchmark identifier into execution-mode and operation names."""

    # A single case parameter avoids the invalid product of all execution modes and operations
    execution_mode, operation = case_name.split("-", maxsplit=1)
    if execution_mode not in ("dask", "multiprocessing") or operation not in OPERATION_BY_NAME:
        raise ValueError(f"Unknown benchmark operation case: {case_name}")
    return execution_mode, operation  # type: ignore[return-value]
