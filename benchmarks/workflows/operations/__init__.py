"""Discover benchmark operations and collect their local specifications."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, cast

from benchmarks.workflows.config import (
    BenchmarkCase,
    BenchmarkConfig,
    Comparison,
    ExecutionMode,
    ExternalReferenceCase,
    Operation,
    OperationCoverage,
    OperationName,
    Sweep,
)


@dataclass(frozen=True)
class OperationCatalog:
    """Collect the operation-local declarations used by runners, ASV and reports."""

    modules: tuple[ModuleType, ...]
    operations: tuple[Operation, ...]
    sweeps: tuple[Sweep, ...]
    comparisons: tuple[Comparison, ...]
    coverage: tuple[OperationCoverage, ...]
    cases: tuple[BenchmarkCase, ...]
    references: tuple[ExternalReferenceCase, ...]


def discover_operation_modules() -> tuple[ModuleType, ...]:
    """Import operation modules in a stable name order."""

    module_names = sorted(module.name for module in pkgutil.iter_modules(__path__) if not module.name.startswith("_"))
    modules = tuple(importlib.import_module(f"{__name__}.{module_name}") for module_name in module_names)
    return tuple(sorted(modules, key=lambda module: (getattr(module, "ORDER", 100), module.__name__)))


def _unique_by(values: Iterable[Any], attribute: str, kind: str) -> tuple[Any, ...]:
    """Return values after rejecting duplicate identifiers."""

    unique: dict[str, Any] = {}
    for value in values:
        identifier = getattr(value, attribute)
        if identifier in unique:
            raise ValueError(f"Duplicate {kind} ID: {identifier}")
        unique[identifier] = value
    return tuple(unique.values())


def collect_operation_modules(modules: Iterable[Any]) -> OperationCatalog:
    """Collect and validate declarations exported by operation modules."""

    ordered_modules = cast(tuple[ModuleType, ...], tuple(modules))
    operations = _unique_by(
        (operation for module in ordered_modules for operation in getattr(module, "OPERATIONS", ())),
        "name",
        "operation",
    )
    sweeps = _unique_by(
        (sweep for module in ordered_modules for sweep in getattr(module, "SWEEPS", ())),
        "id",
        "sweep",
    )
    comparisons = _unique_by(
        (comparison for module in ordered_modules for comparison in getattr(module, "COMPARISONS", ())),
        "slug",
        "comparison",
    )
    coverage = tuple(
        sorted(
            _unique_by(
                (case for module in ordered_modules for case in getattr(module, "COVERAGE", ())),
                "operation",
                "coverage",
            ),
            key=lambda case: case.order,
        )
    )

    # Every generated ASV name must belong to one sweep so setup can resolve its configuration directly
    cases = _unique_by(
        (case for sweep in sweeps for case in sweep.cases),
        "benchmark_class",
        "benchmark case",
    )
    references = _unique_by(
        (reference for sweep in sweeps for reference in sweep.references),
        "benchmark_class",
        "external reference",
    )
    operation_names = {operation.name for operation in operations}
    for case in cases:
        if case.operation not in operation_names:
            raise ValueError(f"Benchmark case {case.benchmark_class} has no operation handler")
    return OperationCatalog(ordered_modules, operations, sweeps, comparisons, coverage, cases, references)


CATALOG = collect_operation_modules(discover_operation_modules())
OPERATION_MODULES = CATALOG.modules
OPERATIONS = CATALOG.operations
SWEEPS = CATALOG.sweeps
COMPARISONS = CATALOG.comparisons
OPERATION_CASES = CATALOG.coverage
BENCHMARK_CASES = CATALOG.cases
EXTERNAL_REFERENCE_CASES = CATALOG.references

OPERATION_BY_NAME = {operation.name: operation for operation in OPERATIONS}
OPERATION_COVERAGE_BY_NAME = {case.operation: case for case in OPERATION_CASES}
SWEEP_BY_ID = {sweep.id: sweep for sweep in SWEEPS}
BENCHMARK_CASE_BY_CLASS = {case.benchmark_class: case for case in BENCHMARK_CASES}
EXTERNAL_REFERENCE_CASE_BY_CLASS = {case.benchmark_class: case for case in EXTERNAL_REFERENCE_CASES}
OPERATION_BENCHMARK_CASES = tuple(
    f"{execution}-{case.operation}" for case in OPERATION_CASES for execution in case.execution_modes
)


def split_operation_case(case_name: str) -> tuple[ExecutionMode, OperationName]:
    """Split one stable fixed-benchmark identifier into execution and operation names."""

    execution, operation = case_name.split("-", maxsplit=1)
    if execution not in ("dask", "multiprocessing") or operation not in OPERATION_BY_NAME:
        raise ValueError(f"Unknown benchmark operation case: {case_name}")
    return cast(ExecutionMode, execution), cast(OperationName, operation)


def resolve_operation_parameters(
    operation: OperationName,
    method: str | None = None,
    calculation_engine: Any | None = None,
    strategy: Any | None = None,
    execution_mode: Any | None = None,
) -> tuple[str | None, Any | None, Any | None]:
    """Resolve one operation's defaults through its local specification."""

    config = BenchmarkConfig(
        operation_method=method,
        calculation_engine=calculation_engine,
        operation_strategy=strategy,
    )
    case = OPERATION_BY_NAME[operation].resolve_case(execution_mode or "eager", config)
    return case.method, case.engine, case.strategy


def format_api_label(operation: OperationName, case: BenchmarkCase) -> str:
    """Format the public call from the same options passed to its execution handler."""

    specification = OPERATION_BY_NAME[operation]
    options = specification.option_builder(case, BenchmarkConfig())
    selected = ((name, options[name]) for name in specification.label_options if name in options)
    arguments = ", ".join(f"{name}={value!r}" for name, value in selected)
    return f"{specification.public_call_name}({arguments})"
