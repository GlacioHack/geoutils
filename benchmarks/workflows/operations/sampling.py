"""Define raster sampling and point-conversion benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    POINT_COUNT_AXIS,
    RASTER_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
    external_case,
)
from benchmarks.workflows.fixtures import read_point_file_sample
from geoutils._dispatch import is_dask_dataframe

ORDER = 80
_POINT_OUTPUT_DRIVERS = ("LAS", "LAZ")

############################
# Operation execution
############################


def sampling_options(case: BenchmarkCase, config: BenchmarkConfig) -> Mapping[str, Any]:
    """Build the public sampling options used for execution."""

    if case.operation == "subsample":
        sample_size = config.subsample_size
    else:
        sample_size = 1 if config.pointcloud_subsample_size is None else config.pointcloud_subsample_size
    return {"subsample": sample_size, "random_state": 42, "force_pixel_offset": "center"}


def run_sampling(runner: Any, case: BenchmarkCase) -> float:
    """Compute one raster sample or complete point conversion."""

    raster = runner.make_raster()
    options = sampling_options(case, runner.config)

    # Set the expected rows for a fixed sample, complete conversion or bounded point conversion
    if case.operation == "subsample":
        expected_count = runner.config.subsample_size
    elif runner.config.pointcloud_subsample_size is None:
        expected_count = runner.config.shape[0] * runner.config.shape[1]
    else:
        expected_count = runner.config.pointcloud_subsample_size

    if runner.backend == "dask":
        points = getattr(raster.rst, case.operation)(**options)
        if not is_dask_dataframe(points) or points.pc.is_loaded:
            raise AssertionError(f"Dask {case.operation}() output must remain lazy before computation.")
        dataframe = points.compute()
        if points.pc.is_loaded:
            raise AssertionError("Computing a Dask result must not load its original point cloud wrapper.")
        if raster._in_memory:
            raise AssertionError("Computing Dask points must not load the source raster.")
        output_count = len(dataframe)
        value = float(dataframe["b1"].iloc[0])
    else:
        points = getattr(raster, case.operation)(**options, mp_config=runner._multiproc_config(case.operation))
        if points.is_loaded:
            raise AssertionError(f"Multiprocessing {case.operation}() output must remain unloaded.")
        runner._last_output_file = str(points.name)
        output_count, value = read_point_file_sample(runner._last_output_file, "b1")
        if points.is_loaded:
            raise AssertionError("Reading the output file must not load its PointCloud wrapper.")
        if raster.is_loaded:
            raise AssertionError("Multiprocessing point conversion must not load its source raster.")

    # Check the requested count and constant source values without retaining the complete raster
    if output_count != expected_count:
        raise AssertionError(f"{case.operation} returned an unexpected number of rows.")
    return value


OPERATIONS = (
    Operation("subsample", run_sampling, sampling_options),
    Operation("to_pointcloud", run_sampling, sampling_options),
)
COVERAGE = (
    OperationCoverage("subsample", ("dask", "multiprocessing"), 1, 8),
    OperationCoverage("to_pointcloud", ("dask", "multiprocessing"), 1, 9),
)


############################
# Cases and input sweeps
############################


def _subsample_config(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the source layout and requested output count."""

    size = 1_000 if pr_check else 2_000
    return {"shape": (size, size), "chunks": (1_000, 1_000), "subsample_size": int(parameter)}


def _pointcloud_config(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the source raster and chunks for complete point conversion."""

    size = int(parameter)
    return {"shape": (size, size), "chunks": (1_000, 1_000)}


_SUBSAMPLE_CASES = execution_cases("subsample-size", "subsample", None, None, executions=("dask", "multiprocessing"))
_POINTCLOUD_CASES = execution_cases(
    "to-pointcloud-raster-size", "to_pointcloud", None, None, executions=("dask", "multiprocessing")
)
_LAS_SUBSAMPLE_CASES = tuple(
    execution_cases(
        "subsample-las-laz-size",
        "subsample",
        None,
        None,
        executions=("multiprocessing",),
        output_driver=driver,
    )[0]
    for driver in _POINT_OUTPUT_DRIVERS
)
_LAS_POINTCLOUD_CASES = tuple(
    execution_cases(
        "to-pointcloud-las-laz-size",
        "to_pointcloud",
        None,
        None,
        executions=("multiprocessing",),
        output_driver=driver,
    )[0]
    for driver in _POINT_OUTPUT_DRIVERS
)
_SUBSAMPLE_REFERENCE = external_case(_SUBSAMPLE_CASES, reference="pdal_cli", pr_check=True)
_POINTCLOUD_REFERENCE = external_case(_POINTCLOUD_CASES, reference="pdal_cli", pr_check=True)
_LAS_SUBSAMPLE_REFERENCES = tuple(
    external_case(
        next(case for case in _LAS_SUBSAMPLE_CASES if case.output_driver == driver),
        reference="pdal_cli",
    )
    for driver in _POINT_OUTPUT_DRIVERS
)
_LAS_POINTCLOUD_REFERENCES = tuple(
    external_case(
        next(case for case in _LAS_POINTCLOUD_CASES if case.output_driver == driver),
        reference="pdal_cli",
    )
    for driver in _POINT_OUTPUT_DRIVERS
)

SWEEPS = (
    Sweep(
        "subsample_size",
        POINT_COUNT_AXIS,
        _subsample_config,
        _SUBSAMPLE_CASES,
        (_SUBSAMPLE_REFERENCE,),
    ),
    Sweep(
        "subsample_size",
        POINT_COUNT_AXIS,
        _subsample_config,
        _LAS_SUBSAMPLE_CASES,
        _LAS_SUBSAMPLE_REFERENCES,
    ),
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _pointcloud_config,
        _POINTCLOUD_CASES,
        (_POINTCLOUD_REFERENCE,),
    ),
    Sweep(
        "raster_size",
        RASTER_AXIS,
        _pointcloud_config,
        _LAS_POINTCLOUD_CASES,
        _LAS_POINTCLOUD_REFERENCES,
    ),
)

############################
# Report comparisons
############################

COMPARISONS = (
    comparison(SWEEPS[0], logarithmic_x=True),
    comparison(SWEEPS[2]),
    comparison(
        SWEEPS[1],
        logarithmic_x=True,
        documentation=False,
        summary=False,
    ),
    comparison(
        SWEEPS[3],
        documentation=False,
        summary=False,
    ),
)
