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

"""Prepare and compute deterministic operations shared by every benchmark suite."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import box

from benchmarks.workflows.registry import (
    CalculationEngine,
    ExecutionMode,
    OperationName,
    OperationStrategyName,
    resolve_operation_parameters,
)
from geoutils._misc import (
    _get_process_mem_mb,
    _prepare_benchmark_process,
    _trim_process_memory,
    import_optional,
)
from geoutils.interface.gridding import GriddingEngine, GriddingMethod
from geoutils.profiler import ProfileMetrics, profile_call


@dataclass
class BenchmarkConfig:
    """Collect deterministic data, chunk, worker and profiling settings."""

    shape: tuple[int, int] = (2048, 2048)
    chunks: tuple[int, int] = (512, 512)
    memory_limit: str = "1GB"
    n_workers: int = 1
    threads_per_worker: int = 1
    gdal_cachemax_mb: int = 64
    profile_interval: float = 0.05
    raster_value: float = 1.0
    subsample_size: int = 2048
    ninterp: int = 2048
    point_partition_size: int = 16
    polygon_regions_per_axis: int = 1
    vector_features_per_axis: int = 1
    point_features_per_axis: int = 5
    operation_method: str | None = None
    calculation_engine: CalculationEngine | None = None
    operation_strategy: OperationStrategyName | None = None
    grid_dist_nodata_pixel: float = float("inf")
    dask_write_batch_size: int = 4
    trim_dask_memory: bool = False
    directory: str | None = None


class ProfiledResult:
    """Expose complete-process memory measurements shared by all benchmark implementations."""

    metrics: ProfileMetrics

    @property
    def peak_process_tree_mem_mb(self) -> float:
        """Return peak aggregate memory for the measured process and its children."""

        peak = self.metrics.peak_process_tree_mem_mb
        if peak is None:
            raise RuntimeError("Process-tree memory was not collected for this benchmark result")
        return peak

    @property
    def process_tree_mem_increase_mb(self) -> float:
        """Return peak memory above the initialized process-tree baseline."""

        if not self.metrics.process_tree_mem_mb:
            raise RuntimeError("Process-tree memory was not collected for this benchmark result")
        baseline = self.metrics.process_tree_mem_mb[0][1]
        return max(0.0, self.peak_process_tree_mem_mb - baseline)


@dataclass
class BenchmarkResult(ProfiledResult):
    """Store one computed result together with memory and worker-health measurements."""

    value: float
    metrics: ProfileMetrics
    worker_pids_before: tuple[int, ...] | dict[str, int] = field(default_factory=tuple)
    worker_pids_after: tuple[int, ...] | dict[str, int] = field(default_factory=tuple)
    dask_worker_baseline_mem_mb: float | None = None
    output_file: str | None = None

    @property
    def worker_restarted(self) -> bool:
        """Whether the backend replaced a worker during the operation."""

        return self.worker_pids_before != self.worker_pids_after


def logical_raster_size_mb(config: BenchmarkConfig) -> float:
    """Return the uncompressed float32 raster size in decimal megabytes."""

    return config.shape[0] * config.shape[1] * np.dtype("float32").itemsize / 1_000_000


def memory_limit_mb(memory_limit: str) -> float:
    """Convert the worker-memory formats used by the benchmark suite to decimal megabytes."""

    # Match the common decimal and binary units accepted by Dask
    value = memory_limit.strip().lower()
    if value.endswith("gib"):
        return float(value[:-3]) * 1024**3 / 1_000_000
    if value.endswith("gb"):
        return float(value[:-2]) * 1000
    if value.endswith("mib"):
        return float(value[:-3]) * 1024**2 / 1_000_000
    if value.endswith("mb"):
        return float(value[:-2])
    return float(value) / 1_000_000


def _tiff_block_size(size: int, requested: int) -> int:
    """Return a valid tiled-GeoTIFF block size no larger than one raster axis."""

    # GeoTIFF tile dimensions must be divisible by sixteen
    block_size = min(size, requested, 512)
    return max(16, block_size // 16 * 16)


def read_raster_center(filename: str) -> float:
    """Read one central output pixel without loading the complete raster."""

    with rio.open(filename) as dataset:
        row = dataset.height // 2
        col = dataset.width // 2
        return float(dataset.read(1, window=rio.windows.Window(col, row, 1, 1))[0, 0])


def _write_constant_raster(filename: str, config: BenchmarkConfig) -> None:
    """Write a deterministic constant raster one storage block at a time."""

    if os.path.exists(filename):
        return

    # Use a real WGS84 extent so reprojection exercises a coordinate transform
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = _tiff_block_size(height, config.chunks[0])
    block_x = _tiff_block_size(width, config.chunks[1])

    # Compression keeps the deterministic constant fixture compact on disk
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=4326,
        transform=transform,
        nodata=-99999,
        tiled=True,
        blockxsize=block_x,
        blockysize=block_y,
        compress="DEFLATE",
        BIGTIFF="IF_NEEDED",
    ) as dst:
        # Allocate only the current storage block instead of the complete raster
        for _, window in dst.block_windows(1):
            block = np.full((int(window.height), int(window.width)), config.raster_value, dtype=np.float32)
            dst.write(block, indexes=1, window=window)


def _write_polygon_raster(filename: str, config: BenchmarkConfig) -> None:
    """Write regularly spaced connected regions for polygonization scenarios."""

    if os.path.exists(filename):
        return
    if config.polygon_regions_per_axis < 1:
        raise ValueError("Polygon regions per axis must be strictly positive")

    # Separate value-one rectangles with nodata so every rectangle is one region
    height, width = config.shape
    transform = rio.transform.from_bounds(7.0, 45.0, 8.0, 46.0, width=width, height=height)
    block_y = _tiff_block_size(height, config.chunks[0])
    block_x = _tiff_block_size(width, config.chunks[1])

    # Stream the patterned raster without allocating the complete benchmark input
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=4326,
        transform=transform,
        nodata=-99999,
        tiled=True,
        blockxsize=block_x,
        blockysize=block_y,
        compress="DEFLATE",
        BIGTIFF="IF_NEEDED",
    ) as dst:
        regions = config.polygon_regions_per_axis
        for _, window in dst.block_windows(1):
            # Pixel phases locate the inner half of every regular grid cell
            row_start = int(window.row_off)
            col_start = int(window.col_off)
            rows = np.arange(row_start, row_start + int(window.height))
            cols = np.arange(col_start, col_start + int(window.width))
            row_phase = ((rows + 0.5) * regions / height) % 1
            col_phase = ((cols + 0.5) * regions / width) % 1
            inside_rows = (row_phase >= 0.25) & (row_phase <= 0.75)
            inside_cols = (col_phase >= 0.25) & (col_phase <= 0.75)
            inside = inside_rows[:, None] & inside_cols[None, :]

            # Nodata gaps keep neighboring rectangles disconnected for both engines
            block = np.full(inside.shape, -99999, dtype=np.float32)
            block[inside] = config.raster_value
            dst.write(block, indexes=1, window=window)


def _write_vector_source(filename: str, features_per_axis: int = 1) -> None:
    """Write regularly spaced polygons used by rasterization and mask scenarios."""

    if os.path.exists(filename):
        return
    if features_per_axis < 1:
        raise ValueError("Vector features per axis must be strictly positive")

    # Leave a regular gap around every feature while retaining one central feature
    x_edges = np.linspace(7.05, 7.95, features_per_axis + 1)
    y_edges = np.linspace(45.05, 45.95, features_per_axis + 1)
    geometries = []
    for x_start, x_stop in zip(x_edges[:-1], x_edges[1:]):
        for y_start, y_stop in zip(y_edges[:-1], y_edges[1:]):
            x_margin = (x_stop - x_start) * 0.2
            y_margin = (y_stop - y_start) * 0.2
            geometries.append(box(x_start + x_margin, y_start + y_margin, x_stop - x_margin, y_stop - y_margin))

    # Constant burn values give every engine the same binary output
    vector = gpd.GeoDataFrame({"value": np.ones(len(geometries), dtype=np.uint8)}, geometry=geometries, crs=4326)
    vector.to_file(filename, driver="GPKG")


def _write_point_source(filename: str, points_per_axis: int = 5) -> None:
    """Write a regular constant-valued point cloud for gridding scenarios."""

    if os.path.exists(filename):
        return
    if points_per_axis < 1:
        raise ValueError("Points per axis must be strictly positive")

    # Keep points away from the exact border so every geometry is unambiguous
    coords_x = np.linspace(7.05, 7.95, points_per_axis)
    coords_y = np.linspace(45.05, 45.95, points_per_axis)
    xx, yy = np.meshgrid(coords_x, coords_y)
    points = gpd.GeoDataFrame(
        {"z": np.ones(xx.size, dtype=np.float64)},
        geometry=gpd.points_from_xy(xx.ravel(), yy.ravel()),
        crs=4326,
    )
    points.to_file(filename, driver="GPKG")


class BenchmarkRunner:
    """Prepare deterministic files and execute one GeoUtils implementation."""

    def __init__(self, backend: ExecutionMode, config: BenchmarkConfig | None = None) -> None:
        """Prepare runner state without starting worker processes."""

        self.backend = backend
        self.config = config or BenchmarkConfig()
        self.cluster: Any | None = None
        self.client: Any | None = None
        self.mp_cluster: Any | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._dask_config_context: Any | None = None
        self._directory: str | None = None
        self._last_output_file: str | None = None

    @property
    def directory(self) -> str:
        """Return the directory containing sources, outputs and spill files."""

        if self._directory is None:
            raise RuntimeError("BenchmarkRunner has not been prepared")
        return self._directory

    @property
    def raster_file(self) -> str:
        """Return the common input raster path."""

        return os.path.join(self.directory, "source-raster.tif")

    @property
    def polygon_raster_file(self) -> str:
        """Return the patterned raster path used by polygonization."""

        return os.path.join(self.directory, "source-polygonize.tif")

    @property
    def vector_file(self) -> str:
        """Return the common input vector path."""

        return os.path.join(self.directory, "source-vector.gpkg")

    @property
    def point_file(self) -> str:
        """Return the common input point-cloud path."""

        return os.path.join(self.directory, "source-points.gpkg")

    def __enter__(self) -> BenchmarkRunner:
        """Prepare sources and start the selected backend."""

        return self.start()

    def __exit__(self, *args: object) -> None:
        """Close workers and temporary files when leaving the context."""

        self.close()

    def prepare_sources(self) -> BenchmarkRunner:
        """Create deterministic source files without starting any workers."""

        # Use the caller directory when results must survive this runner
        if self._directory is None and self.config.directory is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-benchmark-")
            self._directory = self._tmpdir.name
        elif self._directory is None:
            assert self.config.directory is not None
            os.makedirs(self.config.directory, exist_ok=True)
            self._directory = self.config.directory

        # Each fixture is streamed to disk and can be reused by all operations
        _write_constant_raster(self.raster_file, self.config)
        _write_polygon_raster(self.polygon_raster_file, self.config)
        _write_vector_source(self.vector_file, self.config.vector_features_per_axis)
        _write_point_source(self.point_file, self.config.point_features_per_axis)
        return self

    def start(self) -> BenchmarkRunner:
        """Prepare source files and start workers when the implementation needs them."""

        self.prepare_sources()
        if self.backend == "dask":
            self._start_dask()
        elif self.backend == "multiprocessing":
            self._start_multiprocessing()
        return self

    def _start_dask(self) -> None:
        """Start one local Dask cluster with early disk spilling."""

        # Import benchmark-only packages at runtime to keep Dask optional
        dask = import_optional("dask", extra_name="benchmark")
        distributed = import_optional("distributed", extra_name="benchmark")

        # Spill early enough to avoid pausing or replacing the constrained worker
        self._dask_config_context = dask.config.set(
            {
                "temporary-directory": self.directory,
                "distributed.worker.memory.target": 0.45,
                "distributed.worker.memory.spill": 0.60,
                # Leave room for native task workspaces before Dask pauses the worker
                "distributed.worker.memory.pause": 1.20,
                # Peak memory and stable PIDs are asserted directly by the large data test
                "distributed.worker.memory.terminate": False,
            }
        )
        self._dask_config_context.__enter__()

        # Separate worker processes make their memory independent from the client
        self.cluster = distributed.LocalCluster(
            n_workers=self.config.n_workers,
            threads_per_worker=self.config.threads_per_worker,
            processes=True,
            memory_limit=self.config.memory_limit,
            dashboard_address=":0",
            scheduler_kwargs={"dashboard": False},
            local_directory=self.directory,
            env={"GDAL_CACHEMAX": str(self.config.gdal_cachemax_mb)},
        )
        self.client = distributed.Client(self.cluster)

        # Load operation modules before measurement and configure every worker's live GDAL library
        self.client.run(_prepare_benchmark_process, self.config.gdal_cachemax_mb)

    def _start_multiprocessing(self) -> None:
        """Start a real multiprocessing pool with its normal bounded task lifetime."""

        from geoutils.multiproc.cluster import MpCluster

        # Child processes inherit a bounded GDAL block cache before the pool starts
        previous_cachemax = os.environ.get("GDAL_CACHEMAX")
        os.environ["GDAL_CACHEMAX"] = str(self.config.gdal_cachemax_mb)
        try:
            # Rasterio has already initialized GDAL, so set its live cache while workers fork
            with rio.Env(GDAL_CACHEMAX=self.config.gdal_cachemax_mb):
                # Default recycling bounds allocator and native-library caches in long jobs
                self.mp_cluster = MpCluster(conf={"nb_workers": self.config.n_workers})
        finally:
            # Restore the caller environment after all workers have inherited it
            if previous_cachemax is None:
                os.environ.pop("GDAL_CACHEMAX", None)
            else:
                os.environ["GDAL_CACHEMAX"] = previous_cachemax

    def close(self) -> None:
        """Close workers and remove only temporary files owned by this runner."""

        # Close every execution backend before removing its working directory
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.cluster is not None:
            self.cluster.close()
            self.cluster = None
        if self.mp_cluster is not None:
            self.mp_cluster.close()
            self.mp_cluster = None

        # Restore Dask configuration after all distributed processes have stopped
        if self._dask_config_context is not None:
            self._dask_config_context.__exit__(None, None, None)
            self._dask_config_context = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._directory = None

    def worker_pids(self) -> tuple[int, ...] | dict[str, int]:
        """Return stable worker identifiers for the selected backend."""

        if self.backend == "dask":
            if self.client is None:
                return {}
            return {str(address): int(pid) for address, pid in self.client.run(os.getpid).items()}
        if self.backend == "eager" or self.mp_cluster is None:
            return ()
        return self.mp_cluster.worker_pids()

    def make_raster(self, filename: str | None = None) -> Any:
        """Open one prepared raster through the selected GeoUtils interface."""

        # Most operations use the constant source while polygonization uses regions
        source_file = self.raster_file if filename is None else filename

        if self.backend == "dask":
            from geoutils.raster.xr_accessor import open_raster

            # Raster chunks remain lazy until the output values are computed
            return open_raster(source_file, chunks={"y": self.config.chunks[0], "x": self.config.chunks[1]})

        from geoutils import Raster

        # Eager comparisons load the complete input while multiprocessing reads windows
        return Raster(source_file, load_data=self.backend == "eager")

    def run(self, operation: OperationName, *, profile: bool = True) -> BenchmarkResult:
        """Compute one operation while measuring its complete backend."""

        if self.backend == "dask" and self.client is None:
            raise RuntimeError("BenchmarkRunner must be started before running operations")
        if self.backend == "multiprocessing" and self.mp_cluster is None:
            raise RuntimeError("BenchmarkRunner must be started before running operations")
        if self.backend == "eager" and self._directory is None:
            raise RuntimeError("BenchmarkRunner must be started before running operations")

        # Worker identities reveal failures hidden by automatic replacement
        worker_pids_before = self.worker_pids()
        if self.client is not None:
            # Query workers directly because Dask's first monitor sample can predate warm-up
            worker_baseline = sum(float(value) for value in self.client.run(_get_process_mem_mb).values())
        else:
            worker_baseline = None
        if profile:
            value, metrics = profile_call(
                self._execute,
                operation,
                interval=self.config.profile_interval,
                client=self.client,
                dask=self.backend == "dask",
                # One process-tree measurement is comparable across both backends
                include_children=True,
            )
        else:
            value, metrics = profile_call(
                self._execute,
                operation,
                interval=self.config.profile_interval,
                dask=False,
                include_children=False,
            )
        worker_pids_after = self.worker_pids()

        # Return one small value while leaving large output on disk
        return BenchmarkResult(
            value=float(value),
            metrics=metrics,
            worker_pids_before=worker_pids_before,
            worker_pids_after=worker_pids_after,
            dask_worker_baseline_mem_mb=worker_baseline,
            output_file=self._last_output_file,
        )

    def _output_path(self, operation: OperationName, suffix: str = ".tif") -> str:
        """Return one reusable output path for an operation."""

        filename = os.path.join(self.directory, f"output-{self.backend}-{operation}{suffix}")
        if os.path.isfile(filename):
            os.remove(filename)
        return filename

    def _multiproc_config(self, operation: OperationName) -> Any:
        """Build a multiprocessing configuration sharing this runner's worker pool."""

        from geoutils.multiproc import MultiprocConfig

        return MultiprocConfig(
            chunks=self.config.chunks,
            outfile=self._output_path(operation),
            cluster=self.mp_cluster,
        )

    def _write_dask_raster(self, raster: Any, operation: OperationName) -> str:
        """Compute and write one lazy raster in bounded groups of blocks."""

        filename = self._output_path(operation)
        block_y = _tiff_block_size(self.config.shape[0], self.config.chunks[0])
        block_x = _tiff_block_size(self.config.shape[1], self.config.chunks[1])

        # GeoTIFF has no boolean sample type, so masks use the equivalent byte values
        if np.issubdtype(raster.dtype, np.bool_):
            raster = raster.astype("uint8")

        # Read georeferencing from metadata without evaluating any raster value
        data = raster.data
        if data.chunks is None:
            raise ValueError("Dask benchmark output must remain chunked before writing")
        if self.config.dask_write_batch_size < 1:
            raise ValueError("Dask write batch size must be strictly positive")
        nodata = raster.rio.nodata

        # Computing a few blocks together reduces scheduling overhead while retaining a fixed memory bound
        dask = import_optional("dask", extra_name="benchmark")
        pending_blocks = []
        pending_windows: list[rio.windows.Window] = []

        def write_pending_blocks(destination: rio.io.DatasetWriter) -> None:
            """Compute and write the current bounded group of output blocks."""

            if not pending_blocks:
                return

            # One scheduler request computes the independent blocks as a group
            computed_blocks = dask.compute(*pending_blocks)
            for block, window in zip(computed_blocks, pending_windows):
                destination.write(np.asarray(block), indexes=1, window=window)
            pending_blocks.clear()
            pending_windows.clear()

            # Large data contracts may release native workspaces between bounded groups
            if self.config.trim_dask_memory:
                _trim_process_memory()
                if self.client is not None:
                    self.client.run(_trim_process_memory)

        # Open one tiled destination shared by all bounded block groups
        with rio.open(
            filename,
            "w",
            driver="GTiff",
            height=int(data.shape[0]),
            width=int(data.shape[1]),
            count=1,
            dtype=np.dtype(data.dtype),
            crs=raster.rio.crs,
            transform=raster.rio.transform(),
            nodata=nodata,
            tiled=True,
            blockxsize=block_x,
            blockysize=block_y,
            BIGTIFF="IF_NEEDED",
        ) as destination:
            row_offset = 0
            for row_index, row_size in enumerate(data.chunks[0]):
                col_offset = 0
                for col_index, col_size in enumerate(data.chunks[1]):
                    # Retain lazy blocks only until the fixed batch is ready to compute
                    pending_blocks.append(data.blocks[row_index, col_index])
                    window = rio.windows.Window(col_offset, row_offset, col_size, row_size)
                    pending_windows.append(window)
                    if len(pending_blocks) == self.config.dask_write_batch_size:
                        write_pending_blocks(destination)
                    col_offset += col_size
                row_offset += row_size

            # Write a final partial group at the edge of the output raster
            write_pending_blocks(destination)
        return filename

    def _compute_raster(self, raster: Any, operation: OperationName) -> float:
        """Write or inspect the complete raster produced by one implementation."""

        if self.backend == "dask":
            self._last_output_file = self._write_dask_raster(raster, operation)
        elif self.backend == "multiprocessing":
            # Multiprocessing operations already wrote their returned Raster to disk
            self._last_output_file = str(raster.name)
        else:
            # Eager results are already in memory and use the same tiled output contract
            self._last_output_file = self._output_path(operation)
            block_y = _tiff_block_size(self.config.shape[0], self.config.chunks[0])
            block_x = _tiff_block_size(self.config.shape[1], self.config.chunks[1])
            raster.to_file(
                self._last_output_file,
                co_opts={
                    "TILED": "YES",
                    "BLOCKYSIZE": str(block_y),
                    "BLOCKXSIZE": str(block_x),
                    "COMPRESS": "NONE",
                },
            )
        return read_raster_center(self._last_output_file)

    def _interpolation_points(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Create deterministic point coordinates spread across the source raster."""

        # A uniform distribution touches many chunks and avoids incomplete edge support
        rng = np.random.default_rng(42)
        x = rng.uniform(7.01, 7.99, size=self.config.ninterp)
        y = rng.uniform(45.01, 45.99, size=self.config.ninterp)
        return x, y

    def _execute(self, operation: OperationName) -> float:
        """Build and fully compute one named benchmark operation."""

        self._last_output_file = None
        operation_method, calculation_engine, operation_strategy = resolve_operation_parameters(
            operation,
            self.config.operation_method,
            self.config.calculation_engine,
            self.config.operation_strategy,
            self.backend,
        )

        # Open a raster only for operations that use one as their source
        raster: Any = None
        if operation not in ("rasterize", "create_mask", "grid"):
            source_file = self.polygon_raster_file if operation == "polygonize" else self.raster_file
            raster = self.make_raster(source_file)

        if operation == "crop":
            if self.backend != "dask":
                raise ValueError("Deferred raster cropping is only registered for Dask")

            # Crop metadata and array indexes lazily before writing the selected region
            output = raster.rst.crop((7.1, 45.1, 7.9, 45.9))
            return self._compute_raster(output, operation)

        if operation == "translate":
            if self.backend != "dask":
                raise ValueError("Deferred raster translation is only registered for Dask")

            # Translation changes georeferencing while leaving every value chunk deferred
            output = raster.rst.translate(xoff=0.1, yoff=0.1)
            return self._compute_raster(output, operation)

        if operation == "copy":
            if self.backend != "dask":
                raise ValueError("Lazy raster copying is only registered for Dask")

            # A shallow accessor copy duplicates metadata without evaluating the graph
            output = raster.rst.copy(deep=False)
            return self._compute_raster(output, operation)

        if operation == "filter":
            # Apply a local operation before writing its complete large output
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            filter_kwargs: dict[str, Any] = {
                "method": operation_method,
                "engine": calculation_engine,
                "size": 5,
            }
            output = (
                raster.rst.filter(**filter_kwargs)
                if self.backend == "dask"
                else raster.filter(**filter_kwargs, mp_config=mp_config)
            )
            return self._compute_raster(output, operation)

        if operation == "reproject":
            # Fix the target size so GeoUtils and GDAL references write the same pixel count
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            kwargs = {
                "crs": 32632,
                "grid_size": self.config.shape[::-1],
                "resampling": operation_method,
                "nodata": -99999,
                "n_threads": 1,
                "memory_limit": 64,
            }
            output = (
                raster.rst.reproject(**kwargs)
                if self.backend == "dask"
                else raster.reproject(**kwargs, mp_config=mp_config)
            )
            return self._compute_raster(output, operation)

        if operation == "statistics":
            if self.backend != "dask":
                raise ValueError("Raster statistics are only registered for Dask")

            # Compute selected reductions without evaluating unrelated quantiles
            import_optional("dask", extra_name="benchmark")
            import dask

            statistics = raster.rst.get_stats(["mean", "std", "valid count"])
            mean, _, _ = dask.compute(*statistics.values())
            return float(mean)

        if operation == "subsample":
            # Return only a fixed-size selection from the much larger raster
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            subsample_kwargs: dict[str, Any] = {"random_state": 42}
            if operation_strategy is not None:
                subsample_kwargs["strategy"] = operation_strategy
            sample = (
                raster.rst.subsample(
                    self.config.subsample_size,
                    **subsample_kwargs,
                )
                if self.backend == "dask"
                else raster.subsample(
                    self.config.subsample_size,
                    **subsample_kwargs,
                    mp_config=mp_config,
                )
            )
            if hasattr(sample, "compute"):
                sample = sample.compute()
            return float(np.asarray(sample).mean())

        if operation == "interp_points":
            # Interpolate only requested positions while source chunks stay file-backed
            points = self._interpolation_points()
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            values = (
                raster.rst.interp_points(points, method=operation_method, as_array=True)
                if self.backend == "dask"
                else raster.interp_points(points, method=operation_method, as_array=True, mp_config=mp_config)
            )
            if hasattr(values, "compute"):
                values = values.compute()
            return float(np.nanmean(values))

        if operation == "polygonize":
            # The selected chunk strategy reconciles polygons that cross output tiles
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            polygonize_kwargs: dict[str, Any] = {"target_values": 1}
            if operation_strategy is not None:
                polygonize_kwargs["strategy"] = operation_strategy
            polygons = (
                raster.rst.polygonize(**polygonize_kwargs)
                if self.backend == "dask"
                else raster.polygonize(**polygonize_kwargs, mp_config=mp_config)
            )
            self._last_output_file = self._output_path(operation, suffix=".gpkg")
            polygon_data = polygons if isinstance(polygons, gpd.GeoDataFrame) else polygons.ds
            polygon_data.to_file(self._last_output_file)
            return float(len(polygon_data))

        if operation == "write":
            if self.backend != "dask":
                raise ValueError("Direct lazy writing is only registered for Dask")

            # Write the unchanged lazy source to isolate the storage path
            self._last_output_file = self._write_dask_raster(raster, operation)
            return read_raster_center(self._last_output_file)

        if operation in ("rasterize", "create_mask"):
            from geoutils import Vector

            # Vector input is small while the produced raster is larger than memory
            vector = Vector(self.vector_file)
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            kwargs = {
                "shape": self.config.shape,
                "bounds": (7.0, 45.0, 8.0, 46.0),
                "crs": 4326,
                "chunksizes": self.config.chunks,
            }
            if operation == "rasterize":
                # Binary burn values need one byte per pixel and match the GDAL Byte reference
                output = vector.rasterize(
                    in_value=1,
                    out_value=0,
                    out_dtype=np.uint8,
                    dask=self.backend == "dask",
                    mp_config=mp_config,
                    **kwargs,
                )
            else:
                output = vector.create_mask(
                    dask=self.backend == "dask",
                    mp_config=mp_config,
                    **kwargs,
                )
            return self._compute_raster(output, operation)

        if operation == "grid":
            import geoutils as gu

            # Point input and raster output both remain partitioned for their backend
            mp_config = self._multiproc_config(operation) if self.backend == "multiprocessing" else None
            pointcloud = (
                gu.open_pointcloud(
                    self.point_file,
                    data_column="z",
                    chunks=self.config.point_partition_size,
                )
                if self.backend == "dask"
                else gu.PointCloud(self.point_file, data_column="z")
            )
            grid_kwargs = {
                "shape": self.config.shape,
                "bounds": (7.0, 45.0, 8.0, 46.0),
                "resampling": cast(GriddingMethod, operation_method),
                "dist_nodata_pixel": self.config.grid_dist_nodata_pixel,
                "engine": cast(GriddingEngine, calculation_engine),
                "chunksizes": self.config.chunks,
                # One SciPy thread keeps backend and GDAL comparisons repeatable
                "n_threads": 1,
            }
            output = (
                pointcloud.pc.grid(**grid_kwargs)
                if self.backend == "dask"
                else pointcloud.grid(**grid_kwargs, mp_config=mp_config)
            )
            return self._compute_raster(output, operation)

        raise ValueError(f"Unsupported benchmark operation: {operation}")
