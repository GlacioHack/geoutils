"""Automatically collect benchmark operations defined in the ``operations/`` folder."""

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
    Operation,
    OperationCoverage,
    OperationName,
    Sweep,
    comparison,
)

############################
# Automatic discovery
############################


@dataclass(frozen=True)
class OperationCatalog:
    """Collect the operation declarations used by runners, ASV and reports."""

    modules: tuple[ModuleType, ...]
    operations: tuple[Operation, ...]
    sweeps: tuple[Sweep, ...]
    comparisons: tuple[Comparison, ...]
    coverage: tuple[Operation, ...]
    cases: tuple[BenchmarkCase, ...]
    references: tuple[BenchmarkCase, ...]


def discover_operation_modules() -> tuple[ModuleType, ...]:
    """Import operation modules in a stable name order."""

    # Find every operation module
    module_names = sorted(module.name for module in pkgutil.iter_modules(__path__) if not module.name.startswith("_"))

    # Import the declarations, and apply their custom order
    modules = tuple(importlib.import_module(f"{__name__}.{module_name}") for module_name in module_names)
    return tuple(sorted(modules, key=lambda module: (getattr(module, "ORDER", 100), module.__name__)))


def unique_by(values: Iterable[Any], attribute: str, kind: str) -> tuple[Any, ...]:
    unique: dict[str, Any] = {}
    for value in values:
        identifier = getattr(value, attribute)
        if identifier in unique:
            raise ValueError(f"Duplicate {kind} ID: {identifier}")
        unique[identifier] = value

    return tuple(unique.values())


def module_comparisons(module: ModuleType) -> tuple[Comparison, ...]:
    """Return custom plots or one default plot per sweep."""

    # Generate the common one-plot-per-sweep layout unless the operation declares custom groupings
    declared = getattr(module, "COMPARISONS", None)
    return declared if declared is not None else tuple(comparison(sweep) for sweep in getattr(module, "SWEEPS", ()))


def unique_cases(sweeps: tuple[Sweep, ...], *, references: bool = False) -> tuple[BenchmarkCase, ...]:
    """Return cases after rejecting duplicate generated ASV class names."""

    # A class name depends on both the implementation and its parent sweep axis
    unique: dict[str, BenchmarkCase] = {}
    for sweep in sweeps:
        selected = sweep.references if references else sweep.cases
        for case in selected:
            class_name = sweep.benchmark_class(case)
            if class_name in unique:
                raise ValueError(f"Duplicate benchmark case ID: {class_name}")
            unique[class_name] = case

    return tuple(unique.values())


def collect_operation_modules(modules: Iterable[Any]) -> OperationCatalog:
    """Collect declarations from operation modules."""

    # Freeze the discovered order before collecting each kind of declaration
    ordered_modules = cast(tuple[ModuleType, ...], tuple(modules))
    operations = unique_by(
        (operation for module in ordered_modules for operation in getattr(module, "OPERATIONS", ())),
        "name",
        "operation",
    )
    sweeps = unique_by(
        (sweep for module in ordered_modules for sweep in getattr(module, "SWEEPS", ())),
        "id",
        "sweep",
    )

    # Build report comparisons and order the operations used by fixed benchmarks and large-data tests
    comparisons = unique_by(
        (item for module in ordered_modules for item in module_comparisons(module)),
        "slug",
        "comparison",
    )
    coverage = tuple(
        sorted(
            (operation for operation in operations if operation.coverage is not None),
            key=lambda operation: cast(OperationCoverage, operation.coverage).order,
        )
    )

    # Ensure uniqueness
    cases = unique_cases(sweeps)
    references = unique_cases(sweeps, references=True)

    # Check that every benchmark operation is adequately defined
    operation_names = {operation.name for operation in operations}
    for sweep in sweeps:
        if sweep.operation not in operation_names:
            raise ValueError(f"Benchmark sweep {sweep.id!r} has no operation handler")

    # Return catalog shared by ASV, report rendering and large-data tests
    return OperationCatalog(ordered_modules, operations, sweeps, comparisons, coverage, cases, references)


# Discover once at import so ASV receives cases through sweeps and the renderer receives their report comparisons
CATALOG = collect_operation_modules(discover_operation_modules())
OPERATION_MODULES = CATALOG.modules
OPERATIONS = CATALOG.operations
SWEEPS = CATALOG.sweeps
COMPARISONS = CATALOG.comparisons
OPERATION_CASES = CATALOG.coverage
BENCHMARK_CASES = CATALOG.cases
EXTERNAL_REFERENCE_CASES = CATALOG.references

# Index operation and sweep declarations for runner and report lookups
OPERATION_BY_NAME = {operation.name: operation for operation in OPERATIONS}
OPERATION_COVERAGE_BY_NAME = {operation.name: operation.coverage for operation in OPERATION_CASES}
SWEEP_BY_ID = {sweep.id: sweep for sweep in SWEEPS}
BENCHMARK_CASE_BY_CLASS = {sweep.benchmark_class(case): case for sweep in SWEEPS for case in sweep.cases}
EXTERNAL_REFERENCE_CASE_BY_CLASS = {sweep.benchmark_class(case): case for sweep in SWEEPS for case in sweep.references}

# Build the identifiers used by benchmarks and large-data tests
OPERATION_BENCHMARK_CASES = tuple(
    f"{execution}-{operation.name}"
    for operation in OPERATION_CASES
    for execution in cast(OperationCoverage, operation.coverage).execution_modes
)

############################
# Operation lookup helpers
############################


def split_operation_case(case_name: str) -> tuple[ExecutionMode, OperationName]:
    """Split one benchmark identifier into execution and operation names."""

    execution, operation = case_name.split("-", maxsplit=1)

    # Reject names that cannot resolve to one supported worker mode and registered operation
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
    """Resolve operation defaults through its local specification."""

    # Collect the choices in the same config used by runners
    config = BenchmarkConfig(
        operation_method=method,
        calculation_engine=calculation_engine,
        operation_strategy=strategy,
    )

    # Appl defaults and validation for the requested execution mode
    case = OPERATION_BY_NAME[operation].resolve_case(execution_mode or "inmem", config)
    return case.method, case.engine, case.strategy


def format_api_label(case: BenchmarkCase) -> str:
    """Format the public call from the same options passed to its execution handler."""

    # Build the public options once so labels remain aligned with the measured call
    specification = OPERATION_BY_NAME[case.operation]
    options = specification.option_builder(case, BenchmarkConfig())

    # Keep only options chosen by the operation for concise report labels
    selected = ((name, options[name]) for name in specification.label_options if name in options)
    arguments = ", ".join(f"{name}={value!r}" for name, value in selected)
    return f"{specification.public_call_name}({arguments})"
