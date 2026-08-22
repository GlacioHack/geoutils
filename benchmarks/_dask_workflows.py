# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reusable Dask raster workflows for GeoUtils memory stress tests and ASV benchmarks."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import rasterio as rio

from geoutils._misc import import_optional
from geoutils.profiler import ProfileMetrics, profile_call
from geoutils.raster.xr_accessor import RasterAccessor

DaskRasterWorkflow = Literal[
    "filter_reduce",
    "subsample_topk",
    "interp_points",
    "reproject_reduce",
    "write_geotiff",
    "persist_reduce",
]
PolygonizeStrategy = Literal["label_union", "label_stitch", "geometry_stitch"]

DEFAULT_DASK_WORKFLOWS: tuple[DaskRasterWorkflow, ...] = (
    "filter_reduce",
    "subsample_topk",
    "interp_points",
    "reproject_reduce",
    "write_geotiff",
)


@dataclass
class DaskRasterWorkflowConfig:
    """Configuration for large lazy-raster Dask workflows."""

    shape: tuple[int, int] = (4096, 4096)
    chunks: tuple[int, int] = (1024, 1024)
    memory_limit: str = "512MB"
    n_workers: int = 1
    threads_per_worker: int = 1
    spill_directory: str | None = None
    profile_interval: float = 0.05
    raster_value: float = 1.0
    subsample_size: int = 4096
    ninterp: int = 4096
    polygonize_block_size: tuple[int, int] = (96, 128)


@dataclass
class DaskRasterWorkflowResult:
    """Output of a profiled large-Dask workflow."""

    value: float
    metrics: ProfileMetrics
    worker_pids_before: dict[str, int] = field(default_factory=dict)
    worker_pids_after: dict[str, int] = field(default_factory=dict)
    output_file: str | None = None

    @property
    def worker_restarted(self) -> bool:
        """Whether worker process IDs changed during execution."""

        return self.worker_pids_before != self.worker_pids_after


class DaskRasterWorkflowRunner:
    """Context manager for constrained local Dask raster workflows."""

    def __init__(self, config: DaskRasterWorkflowConfig | None = None) -> None:
        self.config = config or DaskRasterWorkflowConfig()
        self.cluster: Any | None = None
        self.client: Any | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._dask_config_context: Any | None = None
        self._spill_directory: str | None = None

    @property
    def spill_directory(self) -> str:
        if self._spill_directory is None:
            raise RuntimeError("DaskRasterWorkflowRunner has not been started.")
        return self._spill_directory

    def __enter__(self) -> DaskRasterWorkflowRunner:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()

    def start(self) -> DaskRasterWorkflowRunner:
        """Start the local distributed cluster."""

        dask = import_optional("dask", extra_name="benchmark")
        distributed = import_optional("distributed", extra_name="benchmark")

        if self.config.spill_directory is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-dask-spill-")
            self._spill_directory = self._tmpdir.name
        else:
            os.makedirs(self.config.spill_directory, exist_ok=True)
            self._spill_directory = self.config.spill_directory

        self._dask_config_context = dask.config.set(
            {
                "temporary-directory": self.spill_directory,
                "distributed.worker.memory.target": 0.30,
                "distributed.worker.memory.spill": 0.40,
                "distributed.worker.memory.pause": 0.80,
                "distributed.worker.memory.terminate": 0.98,
            }
        )
        self._dask_config_context.__enter__()

        self.cluster = distributed.LocalCluster(
            n_workers=self.config.n_workers,
            threads_per_worker=self.config.threads_per_worker,
            processes=True,
            memory_limit=self.config.memory_limit,
            dashboard_address=":0",
            scheduler_kwargs={"dashboard": False},
            local_directory=self.spill_directory,
        )
        self.client = distributed.Client(self.cluster)
        return self

    def close(self) -> None:
        """Close the local distributed cluster and clean temporary directories."""

        if self.client is not None:
            self.client.close()
            self.client = None
        if self.cluster is not None:
            self.cluster.close()
            self.cluster = None
        if self._dask_config_context is not None:
            self._dask_config_context.__exit__(None, None, None)
            self._dask_config_context = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
            self._spill_directory = None

    def make_raster(self) -> Any:
        """Create a lazily generated constant raster."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        data = da.full(
            self.config.shape,
            fill_value=self.config.raster_value,
            chunks=self.config.chunks,
            dtype=np.float32,
        )
        transform = rio.transform.from_origin(0.0, float(self.config.shape[0]), 1.0, 1.0)
        return RasterAccessor.from_array(data=data, transform=transform, crs=3857, nodata=None)

    def make_categorical_raster(self) -> Any:
        """Create a lazy categorical raster with components crossing chunk boundaries."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        block_y, block_x = self.config.polygonize_block_size
        rows = da.arange(self.config.shape[0], chunks=self.config.chunks[0])[:, None]
        cols = da.arange(self.config.shape[1], chunks=self.config.chunks[1])[None, :]
        data = (((rows // block_y) + (cols // block_x)) % 2).astype(np.uint8)
        transform = rio.transform.from_origin(0.0, float(self.config.shape[0]), 1.0, 1.0)
        return RasterAccessor.from_array(data=data, transform=transform, crs=3857, nodata=0)

    def worker_pids(self) -> dict[str, int]:
        """Return Dask worker PIDs keyed by worker address."""

        if self.client is None:
            return {}
        return {str(address): int(pid) for address, pid in self.client.run(os.getpid).items()}

    def run(self, workflow: DaskRasterWorkflow, *, profile: bool = True) -> DaskRasterWorkflowResult:
        """Run a representative large-raster Dask workflow."""

        if self.client is None:
            raise RuntimeError("DaskRasterWorkflowRunner must be used as a context manager or started explicitly.")

        worker_pids_before = self.worker_pids()
        if profile:
            value, metrics = profile_call(
                self._execute_workflow,
                workflow,
                interval=self.config.profile_interval,
                client=self.client,
            )
        else:
            value, metrics = profile_call(
                self._execute_workflow,
                workflow,
                interval=self.config.profile_interval,
                dask=False,
            )
        worker_pids_after = self.worker_pids()
        return DaskRasterWorkflowResult(
            value=float(value),
            metrics=metrics,
            worker_pids_before=worker_pids_before,
            worker_pids_after=worker_pids_after,
            output_file=self._last_output_file if workflow == "write_geotiff" else None,
        )

    def run_polygonize_strategy(
        self, strategy: PolygonizeStrategy, *, profile: bool = True
    ) -> DaskRasterWorkflowResult:
        """Run one Dask-backed polygonize strategy benchmark."""

        if self.client is None:
            raise RuntimeError("DaskRasterWorkflowRunner must be used as a context manager or started explicitly.")

        worker_pids_before = self.worker_pids()
        if profile:
            value, metrics = profile_call(
                self._execute_polygonize_strategy,
                strategy,
                interval=self.config.profile_interval,
                client=self.client,
            )
        else:
            value, metrics = profile_call(
                self._execute_polygonize_strategy,
                strategy,
                interval=self.config.profile_interval,
                dask=False,
            )
        worker_pids_after = self.worker_pids()
        return DaskRasterWorkflowResult(
            value=float(value),
            metrics=metrics,
            worker_pids_before=worker_pids_before,
            worker_pids_after=worker_pids_after,
        )

    def _execute_workflow(self, workflow: DaskRasterWorkflow) -> float:
        self._last_output_file: str | None = None
        raster = self.make_raster()

        if workflow == "filter_reduce":
            filtered = raster.rst.filter(method="mean", size=5)
            return _to_float(filtered.mean().compute())

        if workflow == "subsample_topk":
            sample = raster.rst.subsample(
                subsample=self.config.subsample_size,
                random_state=42,
                strategy="topk",
            )
            return _to_float(sample.compute() if hasattr(sample, "compute") else sample)

        if workflow == "interp_points":
            x, y = _interp_points(shape=self.config.shape, npoints=self.config.ninterp)
            interpolated = raster.rst.interp_points(points=(x, y), method="linear", as_array=True)
            return _to_float(interpolated.compute() if hasattr(interpolated, "compute") else interpolated)

        if workflow == "reproject_reduce":
            reprojected = raster.rst.reproject(crs=3857, res=2.0, resampling="nearest")
            return _to_float(reprojected.mean().compute())

        if workflow == "write_geotiff":
            filtered = raster.rst.filter(method="mean", size=3)
            self._last_output_file = os.path.join(self.spill_directory, f"geoutils-dask-write-{uuid.uuid4()}.tif")
            writer = filtered.rio.to_raster(
                self._last_output_file,
                lock=True,
                compute=False,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            writer.compute()
            with rio.open(self._last_output_file) as ds:
                value = ds.read(1, window=rio.windows.Window(0, 0, 1, 1))[0, 0]
            return float(value)

        if workflow == "persist_reduce":
            if self.client is None:
                raise RuntimeError("DaskRasterWorkflowRunner must be started before running workflows.")
            persisted = self.client.persist(raster.data + self.config.raster_value)
            import_optional("distributed", extra_name="benchmark").wait(persisted)
            try:
                return _to_float(persisted.mean().compute())
            finally:
                del persisted

        raise ValueError(f"Unsupported Dask raster workflow: {workflow}.")

    def _execute_polygonize_strategy(self, strategy: PolygonizeStrategy) -> float:
        raster = self.make_categorical_raster()
        polygons = raster.rst.polygonize(target_values=1, connectivity=4, strategy=strategy)
        return float(len(polygons))


def _to_float(value: Any) -> float:
    """Convert a small computed result to float."""

    return float(np.asarray(value).mean())


def _interp_points(shape: tuple[int, int], npoints: int) -> tuple[np.ndarray[Any, np.dtype[np.floating[Any]]], ...]:
    """Create deterministic in-bounds interpolation coordinates for the synthetic raster grid."""

    rng = np.random.default_rng(42)
    height, width = shape
    x = rng.uniform(2.0, max(3.0, width - 2.0), size=npoints)
    y = rng.uniform(2.0, max(3.0, height - 2.0), size=npoints)
    return x, y


def profile_dask_raster_workflow(
    workflow: DaskRasterWorkflow,
    config: DaskRasterWorkflowConfig | None = None,
) -> DaskRasterWorkflowResult:
    """Run one profiled large-raster workflow on a constrained local distributed cluster."""

    with DaskRasterWorkflowRunner(config=config) as runner:
        return runner.run(workflow=workflow, profile=True)
