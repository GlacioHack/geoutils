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

"""Prepare shared benchmark infrastructure and run operation-local handlers."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import rasterio as rio

from benchmarks.workflows.config import (
    BenchmarkConfig,
    BenchmarkResult,
    ExecutionMode,
    OperationName,
    ProfiledResult,
    process_tree_memory_increase_mb,
)
from benchmarks.workflows.fixtures import (
    read_raster_center,
)
from benchmarks.workflows.fixtures import (
    tiff_block_size as _tiff_block_size,
)
from benchmarks.workflows.fixtures import (
    write_constant_raster as _write_constant_raster,
)
from benchmarks.workflows.fixtures import (
    write_point_source as _write_point_source,
)
from benchmarks.workflows.fixtures import (
    write_polygon_raster as _write_polygon_raster,
)
from benchmarks.workflows.fixtures import (
    write_vector_source as _write_vector_source,
)
from benchmarks.workflows.operations import OPERATION_BY_NAME
from geoutils._misc import (
    _get_process_mem_mb,
    _prepare_benchmark_process,
    _trim_process_memory,
    import_optional,
)
from geoutils.profiler import profile_call

############################################
# Worker lifecycle and complete operations
############################################


# Prepare the shared inputs, start the selected execution mode and force each operation to produce a complete output
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
            # Rasterio's integer option is measured in bytes; worker environment strings remain measured in MiB
            with rio.Env(GDAL_CACHEMAX=self.config.gdal_cachemax_mb * 1024**2):
                # Default recycling bounds allocator and native-library caches in long jobs
                self.mp_cluster = MpCluster(conf={"nb_workers": self.config.n_workers})

                # Forkserver and spawn workers import modules independently, so finish their warm-up before profiling
                warmup = self.mp_cluster.submit(_prepare_benchmark_process, self.config.gdal_cachemax_mb)
                self.mp_cluster.compute(warmup)
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

        point_output = operation in ("subsample", "to_pointcloud")
        suffix = f".{self.config.point_output_driver.lower()}" if point_output else ".tif"
        return MultiprocConfig(
            chunks=self.config.chunks,
            outfile=self._output_path(operation, suffix=suffix),
            driver=self.config.point_output_driver if point_output else None,
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

    def _execute(self, operation: OperationName) -> float:
        """Resolve and fully compute one operation through its local handler."""

        self._last_output_file = None
        specification = OPERATION_BY_NAME.get(operation)
        if specification is None:
            raise ValueError(f"Unsupported benchmark operation: {operation}")
        case = specification.resolve_case(self.backend, self.config)
        return specification.handler(self, case)
