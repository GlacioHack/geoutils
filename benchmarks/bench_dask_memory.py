"""ASV benchmarks for Dask raster memory scalability."""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np

from benchmarks._dask_workflows import (
    DaskRasterWorkflow,
    DaskRasterWorkflowConfig,
    DaskRasterWorkflowRunner,
    PolygonizeStrategy,
)


class DaskRasterMemory:
    """Large lazy-raster Dask workflows with worker memory metrics."""

    timeout = 900
    param_names = ["workflow"]
    params = [
        [
            "filter_reduce",
            "subsample_topk",
            "interp_points",
            "reproject_reduce",
            "write_geotiff",
        ]
    ]

    def setup(self, workflow: DaskRasterWorkflow) -> None:
        self.tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-dask-")
        self.runner = DaskRasterWorkflowRunner(
            DaskRasterWorkflowConfig(
                shape=(4096, 4096),
                chunks=(1024, 1024),
                memory_limit="1GB",
                spill_directory=self.tmpdir.name,
                profile_interval=0.1,
                subsample_size=2048,
                ninterp=2048,
            )
        )
        self.runner.start()

    def teardown(self, workflow: DaskRasterWorkflow) -> None:
        self.runner.close()
        self.tmpdir.cleanup()

    def time_workflow(self, workflow: DaskRasterWorkflow) -> None:
        self.runner._execute_workflow(workflow)

    def track_runtime_s(self, workflow: DaskRasterWorkflow) -> float:
        return self.runner.run(workflow).metrics.runtime_s

    def track_peak_client_rss_mb(self, workflow: DaskRasterWorkflow) -> float:
        return self.runner.run(workflow).metrics.peak_client_rss_mb

    def track_peak_dask_worker_process_rss_mb(self, workflow: DaskRasterWorkflow) -> float:
        peak = self.runner.run(workflow).metrics.peak_dask_worker_process_rss_mb
        return float("nan") if peak is None else peak

    def track_peak_dask_spilled_mb(self, workflow: DaskRasterWorkflow) -> float:
        peak = self.runner.run(workflow).metrics.peak_dask_spilled_mb
        return float("nan") if peak is None else peak


class DaskSpillPressure:
    """Focused ASV benchmark for Dask spilling under managed-memory pressure."""

    timeout = 900

    def setup(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-dask-spill-")
        self.runner = DaskRasterWorkflowRunner(
            DaskRasterWorkflowConfig(
                shape=(8192, 8192),
                chunks=(1024, 1024),
                memory_limit="512MB",
                spill_directory=self.tmpdir.name,
                profile_interval=0.1,
            )
        )
        self.runner.start()

    def teardown(self) -> None:
        self.runner.close()
        self.tmpdir.cleanup()

    def time_persist_reduce(self) -> None:
        self.runner._execute_workflow("persist_reduce")

    def track_peak_client_rss_mb(self) -> float:
        return self.runner.run("persist_reduce").metrics.peak_client_rss_mb

    def track_peak_dask_worker_process_rss_mb(self) -> float:
        peak = self.runner.run("persist_reduce").metrics.peak_dask_worker_process_rss_mb
        return float("nan") if peak is None else peak

    def track_peak_dask_spilled_mb(self) -> float:
        peak = self.runner.run("persist_reduce").metrics.peak_dask_spilled_mb
        return float("nan") if peak is None else peak


class DaskPolygonizeStrategies:
    """Compare implemented chunked polygonize strategies on the same lazy categorical raster."""

    timeout = 900
    param_names = ["strategy"]
    params = [["label_union", "label_stitch", "geometry_stitch"]]

    def setup(self, strategy: PolygonizeStrategy) -> None:
        self.runner = DaskRasterWorkflowRunner(
            DaskRasterWorkflowConfig(
                shape=(128, 128),
                chunks=(64, 64),
                polygonize_block_size=(40, 48),
            )
        )

    def time_polygonize_strategy(self, strategy: PolygonizeStrategy) -> None:
        self.runner._execute_polygonize_strategy(strategy)

    def track_runtime_s(self, strategy: PolygonizeStrategy) -> float:
        _, metrics = self._profile(strategy)
        return metrics.runtime_s

    def track_peak_client_rss_mb(self, strategy: PolygonizeStrategy) -> float:
        _, metrics = self._profile(strategy)
        return metrics.peak_client_rss_mb

    def track_output_feature_count(self, strategy: PolygonizeStrategy) -> float:
        return self.runner._execute_polygonize_strategy(strategy)

    def _profile(self, strategy: PolygonizeStrategy) -> tuple[float, Any]:
        from geoutils.profiler import profile_call

        return profile_call(self.runner._execute_polygonize_strategy, strategy, dask=False)


class RasterRuntime:
    """Representative high-level CPU/runtime benchmarks on in-memory rasters."""

    timeout = 120

    def setup(self) -> None:
        import geoutils as gu

        self.raster = gu.Raster(gu.examples.get_path_test("everest_landsat_b4"))
        self.raster.load()

    def time_get_stats(self) -> None:
        self.raster.get_stats(["mean", "std", "valid count"])

    def time_filter_mean(self) -> None:
        self.raster.filter(method="mean", size=5)

    def time_subsample_topk(self) -> None:
        self.raster.subsample(2048, random_state=42, strategy="topk")

    def time_reproject_downsample(self) -> None:
        self.raster.reproject(crs=self.raster.crs, res=float(np.mean(self.raster.res)) * 2, resampling="nearest")
