"""Define raster sampling and point-conversion benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.workflows.config import (
    RASTER_AXIS,
    SUBSAMPLE_AXIS,
    BenchmarkCase,
    BenchmarkConfig,
    Operation,
    OperationCoverage,
    Parameter,
    Sweep,
    comparison,
    execution_cases,
    external_case,
    raster_size_config,
)
from benchmarks.workflows.fixtures import read_point_file_sample
from geoutils._dispatch import is_dask_dataframe

ORDER = 80
POINT_OUTPUT_DRIVERS = ("LAS", "LAZ")

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
    Operation("subsample", run_sampling, sampling_options, coverage=OperationCoverage(8)),
    Operation("to_pointcloud", run_sampling, sampling_options, coverage=OperationCoverage(9)),
)


############################
# Cases and input sweeps
############################


def subsample_config(parameter: Parameter, case: BenchmarkCase, pr_check: bool) -> Mapping[str, Any]:
    """Set the source layout and requested output count."""

    size = 1_000 if pr_check else 2_000
    return {"shape": (size, size), "chunks": (1_000, 1_000), "subsample_size": int(parameter)}


# Each case fixes one execution mode and output format for one ASV result series; its sweep owns the changing input
SUBSAMPLE_CASES = execution_cases("subsample", None, None, executions=("dask", "multiprocessing"))
POINTCLOUD_CASES = execution_cases(
    "to_pointcloud",
    None,
    None,
    executions=("dask", "multiprocessing"),
)
LAS_SUBSAMPLE_CASES = tuple(
    BenchmarkCase(
        "subsample",
        execution="multiprocessing",
        output_driver=driver,
    )
    for driver in POINT_OUTPUT_DRIVERS
)
LAS_POINTCLOUD_CASES = tuple(
    BenchmarkCase(
        "to_pointcloud",
        execution="multiprocessing",
        output_driver=driver,
    )
    for driver in POINT_OUTPUT_DRIVERS
)
SUBSAMPLE_REFERENCE = external_case(SUBSAMPLE_CASES, reference="pdal_cli", pr_check=True)
POINTCLOUD_REFERENCE = external_case(POINTCLOUD_CASES, reference="pdal_cli", pr_check=True)
LAS_SUBSAMPLE_REFERENCES = tuple(external_case(case, reference="pdal_cli") for case in LAS_SUBSAMPLE_CASES)
LAS_POINTCLOUD_REFERENCES = tuple(external_case(case, reference="pdal_cli") for case in LAS_POINTCLOUD_CASES)

# Measure every GeoUtils case and matching PDAL reference over output count or raster size
SWEEPS = (
    Sweep(SUBSAMPLE_AXIS, subsample_config, SUBSAMPLE_CASES, (SUBSAMPLE_REFERENCE,)),
    Sweep(
        SUBSAMPLE_AXIS,
        subsample_config,
        LAS_SUBSAMPLE_CASES,
        LAS_SUBSAMPLE_REFERENCES,
        name="subsample-las-laz",
    ),
    Sweep(
        RASTER_AXIS,
        raster_size_config,
        POINTCLOUD_CASES,
        (POINTCLOUD_REFERENCE,),
    ),
    Sweep(
        RASTER_AXIS,
        raster_size_config,
        LAS_POINTCLOUD_CASES,
        LAS_POINTCLOUD_REFERENCES,
        name="to-pointcloud-las-laz",
    ),
)

############################
# Report comparisons
############################

# Comparisons select saved GeoUtils and PDAL series for plots by operation and format; they run no new measurements
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
