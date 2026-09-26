"""Define raster sampling and point-conversion benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from rasterio.transform import from_origin

import geoutils as gu
from benchmarks.asv_suite import asv_parameter_values
from benchmarks.comparisons.dask import topk_indices, topk_keys
from benchmarks.workflows.config import (
    DASK_CUTOFF_SIZES,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_POINT_COUNT,
    DEFAULT_RASTER_SIZE,
    RASTER_SIZES,
    SUBSAMPLE_SIZES,
    Parameter,
    RuntimeConfig,
    raster_size_config,
)
from benchmarks.workflows.core import (
    Case,
    Operation,
    comparison,
    execution_cases,
    parameter_config,
    reference_case,
)
from benchmarks.workflows.io import read_point_file_sample, write_constant_raster
from geoutils._dispatch import is_dask_dataframe
from geoutils._misc import import_optional
from geoutils.profiler import profile_call
from geoutils.sampling.subsampling import _subsample as _subsample_values

ORDER = 80
POINT_OUTPUT_DRIVERS = ("LAS", "LAZ")

########################################
# Define setup for sampling operations
########################################


def sampling_options(case: Case, config: RuntimeConfig) -> Mapping[str, Any]:
    """Build the public sampling options used for execution."""

    if config.value("operation") == "subsample":
        sample_size = int(config.value("subsample_size", DEFAULT_POINT_COUNT))
    else:
        pointcloud_size = config.value("pointcloud_subsample_size")
        sample_size = 1 if pointcloud_size is None else pointcloud_size
    return {"subsample": sample_size, "random_state": 42, "force_pixel_offset": "center"}


def prepare_sampling(runner: Any, case: Case) -> None:
    """Write the raster sampled by GeoUtils and PDAL cases."""

    write_constant_raster(runner.path("source-raster.tif"), runner.config)


def run_sampling(runner: Any, case: Case) -> float:
    """Compute one raster sample or complete point conversion."""

    if case.implementation == "pdal":
        from benchmarks.comparisons.pdal import execute_pdal

        return execute_pdal(runner, case)

    operation = runner.operation.name
    raster = runner.make_raster()
    options = sampling_options(case, runner.config)

    # Set the expected rows for a fixed sample, complete conversion or bounded point conversion
    if operation == "subsample":
        expected_count = int(runner.config.value("subsample_size", DEFAULT_POINT_COUNT))
    elif runner.config.value("pointcloud_subsample_size") is None:
        expected_count = runner.config.shape[0] * runner.config.shape[1]
    else:
        expected_count = int(runner.config.value("pointcloud_subsample_size"))

    if runner.backend == "dask":
        points = getattr(raster.rst, operation)(**options)
        if not is_dask_dataframe(points) or points.pc.is_loaded:
            raise AssertionError(f"Dask {operation}() output must remain lazy before computation.")
        dataframe = points.compute()
        if points.pc.is_loaded:
            raise AssertionError("Computing a Dask result must not load its original point cloud wrapper.")
        if raster._in_memory:
            raise AssertionError("Computing Dask points must not load the source raster.")
        output_count = len(dataframe)
        value = float(dataframe["b1"].iloc[0])
    else:
        suffix = f".{case.output_driver.lower()}"
        points = getattr(raster, operation)(
            **options,
            mp_config=runner._multiproc_config(suffix=suffix, driver=case.output_driver),
        )
        if points.is_loaded:
            raise AssertionError(f"Multiprocessing {operation}() output must remain unloaded.")
        runner._last_output_file = str(points.name)
        output_count, value = read_point_file_sample(runner._last_output_file, "b1")
        if points.is_loaded:
            raise AssertionError("Reading the output file must not load its PointCloud wrapper.")
        if raster.is_loaded:
            raise AssertionError("Multiprocessing point conversion must not load its source raster.")

    # Check the requested count and constant source values without retaining the complete raster
    if output_count != expected_count:
        raise AssertionError(f"{operation} returned an unexpected number of rows.")
    return value


SUBSAMPLE = Operation(
    "subsample",
    prepare_sampling,
    run_sampling,
    sampling_options,
    label="Subsampling",
    order=8,
    large_data_cases=tuple(
        Case(execution=execution, options={"operation": "subsample", "subsample_size": DEFAULT_POINT_COUNT})
        for execution in ("dask", "multiprocessing")
    ),
    large_data_dependencies=("dask_geopandas",),
)
TO_POINTCLOUD = Operation(
    "to_pointcloud",
    prepare_sampling,
    run_sampling,
    sampling_options,
    label="Point cloud conversion",
    order=9,
    large_data_cases=tuple(
        Case(execution=execution, options={"operation": "to_pointcloud", "pointcloud_subsample_size": None})
        for execution in ("dask", "multiprocessing")
    ),
    large_data_dependencies=("dask_geopandas",),
)
OPERATIONS = (SUBSAMPLE, TO_POINTCLOUD)


###################
# Define benchmarks
###################


def subsample_config(parameter: Parameter | None, case: Case) -> Mapping[str, Any]:
    """Set the source layout and requested output count."""

    assert parameter is not None
    return {
        "shape": (DEFAULT_RASTER_SIZE, DEFAULT_RASTER_SIZE),
        "chunks": (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_SIZE),
        "subsample_size": int(parameter),
    }


# Each case fixes one execution mode and output format for one ASV result series; its sweep owns the changing input
SUBSAMPLE_CASES = execution_cases(
    None,
    None,
    executions=("dask", "multiprocessing"),
    options={"operation": "subsample"},
)
POINTCLOUD_CASES = execution_cases(
    None,
    None,
    executions=("dask", "multiprocessing"),
    options={"operation": "to_pointcloud"},
)
LAS_SUBSAMPLE_CASES = tuple(
    Case(
        execution="multiprocessing",
        output_driver=driver,
        options={"operation": "subsample"},
    )
    for driver in POINT_OUTPUT_DRIVERS
)
LAS_POINTCLOUD_CASES = tuple(
    Case(
        execution="multiprocessing",
        output_driver=driver,
        options={"operation": "to_pointcloud"},
    )
    for driver in POINT_OUTPUT_DRIVERS
)
SUBSAMPLE_REFERENCE = reference_case(SUBSAMPLE_CASES, implementation="pdal")
POINTCLOUD_REFERENCE = reference_case(POINTCLOUD_CASES, implementation="pdal")
LAS_SUBSAMPLE_REFERENCES = tuple(reference_case(case, implementation="pdal") for case in LAS_SUBSAMPLE_CASES)
LAS_POINTCLOUD_REFERENCES = tuple(reference_case(case, implementation="pdal") for case in LAS_POINTCLOUD_CASES)


def sampling_workload(parameter: Parameter, configs: tuple[RuntimeConfig, ...]) -> str:
    """Describe the raster, chunks and requested output point count."""

    config = configs[0]
    return (
        f"{config.shape[0]:,} × {config.shape[1]:,} raster; "
        f"{config.chunks[0]:,} × {config.chunks[1]:,} chunks; {int(parameter):,} output points"
    )


# Measure every GeoUtils case and matching PDAL reference over output count or raster size
BENCHMARKS = (
    parameter_config(
        "subsample_size",
        SUBSAMPLE_SIZES,
        SUBSAMPLE,
        (*SUBSAMPLE_CASES, SUBSAMPLE_REFERENCE),
        subsample_config,
        parameter_label="Number of output points",
        parameter_title="output point count",
        describe_workload=sampling_workload,
    ),
    parameter_config(
        "subsample_size",
        SUBSAMPLE_SIZES,
        SUBSAMPLE,
        (*LAS_SUBSAMPLE_CASES, *LAS_SUBSAMPLE_REFERENCES),
        subsample_config,
        name="subsample-las-laz",
        parameter_label="Number of output points",
        parameter_title="output point count",
        describe_workload=sampling_workload,
    ),
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        TO_POINTCLOUD,
        (*POINTCLOUD_CASES, POINTCLOUD_REFERENCE),
        raster_size_config,
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    ),
    parameter_config(
        "raster_size",
        RASTER_SIZES,
        TO_POINTCLOUD,
        (*LAS_POINTCLOUD_CASES, *LAS_POINTCLOUD_REFERENCES),
        raster_size_config,
        name="to-pointcloud-las-laz",
        parameter_label="Size of raster (pixels per side)",
        parameter_title="raster size",
    ),
)

#############################
# Define report comparisons
#############################

# Comparisons select saved GeoUtils and PDAL series for plots by operation and format; they run no new measurements
COMPARISONS = (
    comparison(BENCHMARKS[0], by="execution", logarithmic_x=True),
    comparison(BENCHMARKS[2], by="execution"),
    comparison(
        BENCHMARKS[1],
        by="output_driver",
        logarithmic_x=True,
        documentation=False,
        summary=False,
    ),
    comparison(
        BENCHMARKS[3],
        by="output_driver",
        documentation=False,
        summary=False,
    ),
)


###################################
# Internal sampling microbenchmarks
###################################


class DaskTopkComparison:
    """Measure GeoUtils and Dask top-k selection with the same deterministic cell keys."""

    timeout = 900
    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0

    param_names = ["implementation", "subsample_size"]
    params = [
        ["geoutils", "dask_argtopk"],
        asv_parameter_values(SUBSAMPLE_SIZES),
    ]

    def setup(self, implementation: Literal["geoutils", "dask_argtopk"], subsample_size: int) -> None:
        """Prepare one lazy raster containing regular nodata cells."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        shape = (2048, 2048)
        rows = da.arange(shape[0], chunks=500)[:, None]
        columns = da.arange(shape[1], chunks=500)[None, :]
        positions = rows * shape[1] + columns
        values = da.where(positions % 19 == 0, np.nan, 1.0).astype(np.float32)
        self.raster = gu.RasterAccessor.from_array(values, from_origin(0, shape[0], 1, 1), 32633)

    def _run(self, implementation: Literal["geoutils", "dask_argtopk"], subsample_size: int) -> None:
        """Compute selected cell numbers through one top-k implementation."""

        import dask

        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils":
                rows, columns = _subsample_values(
                    self.raster.rst,
                    subsample_size,
                    return_indices=True,
                    random_state=42,
                    strategy="topk",
                )
                dask.compute(rows, columns)
                return

            # Apply the independent native Dask reduction to the same prepared values
            topk_indices(self.raster.data, subsample_size)

    def time_topk(self, implementation: Literal["geoutils", "dask_argtopk"], subsample_size: int) -> None:
        """Measure complete deterministic selection after preparing the lazy raster."""

        self._run(implementation, subsample_size)

    def track_peak_client_mem_mb(
        self, implementation: Literal["geoutils", "dask_argtopk"], subsample_size: int
    ) -> float:
        """Measure peak client memory during deterministic selection."""

        _, metrics = profile_call(self._run, implementation, subsample_size, dask=False, include_children=False)
        return metrics.peak_client_mem_mb


setattr(DaskTopkComparison.track_peak_client_mem_mb, "unit", "MB")
setattr(
    DaskTopkComparison.time_topk,
    "benchmark_name",
    "asv_suite.subsampling.DaskTopkComparison.time_topk",
)
setattr(
    DaskTopkComparison.track_peak_client_mem_mb,
    "benchmark_name",
    "asv_suite.subsampling.DaskTopkComparison.track_peak_client_mem_mb",
)


class DaskCutoffComparison:
    """Measure the bounded cutoff search against Dask top-k above one raster chunk."""

    timeout = 900
    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0

    param_names = ["implementation", "subsample_size"]
    params = [
        ["geoutils_cutoff", "dask_topk"],
        asv_parameter_values(DASK_CUTOFF_SIZES),
    ]

    def setup(self, implementation: Literal["geoutils_cutoff", "dask_topk"], subsample_size: int) -> None:
        """Prepare equivalent lazy keys and raster chunks for both selection methods."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        shape = (2048, 2048)
        rows = da.arange(shape[0], chunks=500)[:, None]
        columns = da.arange(shape[1], chunks=500)[None, :]
        positions = rows * shape[1] + columns
        self.values = da.where(positions % 19 == 0, np.nan, 1.0).astype(np.float32)
        self.shape = shape

        row_chunks, column_chunks = self.values.chunks
        row_starts = np.cumsum((0, *row_chunks))
        column_starts = np.cumsum((0, *column_chunks))
        tiles = np.array(
            [
                (row_starts[row], row_starts[row + 1], column_starts[column], column_starts[column + 1])
                for row in range(len(row_chunks))
                for column in range(len(column_chunks))
            ],
            dtype=np.int64,
        )
        self.blocks = self.values.to_delayed().ravel().tolist()
        self.block_ids = [
            {
                "row_start": int(tile[0]),
                "row_stop": int(tile[1]),
                "col_start": int(tile[2]),
                "col_stop": int(tile[3]),
            }
            for tile in tiles
        ]
        self.largest_chunk = max(int(rows * columns) for rows in row_chunks for columns in column_chunks)

    def _run(self, implementation: Literal["geoutils_cutoff", "dask_topk"], subsample_size: int) -> None:
        """Find the exact selection boundary with GeoUtils or Dask's native reduction."""

        import dask

        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils_cutoff":
                from geoutils.sampling.subsampling import (
                    SubsampleMeta,
                    _dask_array_topk_cutoff,
                )

                subsample_meta = _dask_array_topk_cutoff(
                    blocks=self.blocks,
                    mask_blocks=[None] * len(self.blocks),
                    block_ids=self.block_ids,
                    array_shape=self.shape,
                    largest_chunk=self.largest_chunk,
                    subsample=subsample_size,
                    subsample_meta=SubsampleMeta(sample_size=subsample_size, seed=42, cutoff=None),
                    skip_nodata=True,
                )
                if subsample_meta.sample_size != subsample_size or subsample_meta.cutoff is None:
                    raise AssertionError("The cutoff search did not find the requested selection boundary.")
                return

            topk_keys(self.values, subsample_size)

    def time_cutoff(self, implementation: Literal["geoutils_cutoff", "dask_topk"], subsample_size: int) -> None:
        """Measure exact selection above the largest raster chunk."""

        self._run(implementation, subsample_size)

    def track_peak_client_mem_mb(
        self, implementation: Literal["geoutils_cutoff", "dask_topk"], subsample_size: int
    ) -> float:
        """Measure peak client memory while finding the exact selection boundary."""

        _, metrics = profile_call(self._run, implementation, subsample_size, dask=False, include_children=False)
        return metrics.peak_client_mem_mb


setattr(DaskCutoffComparison.track_peak_client_mem_mb, "unit", "MB")
setattr(
    DaskCutoffComparison.time_cutoff,
    "benchmark_name",
    "asv_suite.subsampling.DaskCutoffComparison.time_cutoff",
)
setattr(
    DaskCutoffComparison.track_peak_client_mem_mb,
    "benchmark_name",
    "asv_suite.subsampling.DaskCutoffComparison.track_peak_client_mem_mb",
)
