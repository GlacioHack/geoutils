"""Run GeoUtils components beside independent native Dask references."""

from __future__ import annotations

from typing import Literal

import numpy as np
from rasterio.transform import from_origin

import geoutils as gu
from benchmarks.asv_suite import asv_pr_check_enabled
from benchmarks.dask_comparison.reference import global_statistics, topk_indices, topk_keys
from benchmarks.workflows.config import DASK_CUTOFF_AXIS, POINT_COUNT_AXIS, RASTER_AXIS
from geoutils._misc import import_optional
from geoutils.profiler import profile_call
from geoutils.sampling.subsampling import _subsample as _subsample_values
from geoutils.stats.reduction import (
    _normalize_statistics,
    _reduce_values,
)


class GlobalDaskReduction:
    """Measure the shared GeoUtils reducer and native Dask reductions on the same prepared array."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    timeout = 300
    param_names = ["implementation", "raster_size"]
    params = [["geoutils", "native_dask"], list(RASTER_AXIS.parameters(asv_pr_check_enabled()))]

    def setup(self, implementation: Literal["geoutils", "native_dask"], raster_size: int) -> None:
        """Prepare one finite Dask raster and the same mergeable statistic request for both implementations."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        del implementation
        generator = np.random.default_rng(42)
        values = generator.normal(size=(raster_size, raster_size))
        values[::97, ::89] = np.nan
        self.values = da.from_array(values, chunks=(500, 500))
        self.aliases = {"mean", "min", "max", "sum", "sumofsquares", "rmse", "std"}
        self.statistics = _normalize_statistics(sorted(self.aliases), grouped=False)

    def time_reduction(self, implementation: Literal["geoutils", "native_dask"], raster_size: int) -> None:
        """Compute the requested global estimates with the selected Dask reduction implementation."""

        import dask

        del raster_size
        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils":
                _reduce_values([self.values], self.statistics)
            else:
                global_statistics(self.values)


# Keep the stored identifier while locating this internal comparison outside the public ASV modules
setattr(
    GlobalDaskReduction.time_reduction,
    "benchmark_name",
    "asv_suite.global_stats.GlobalDaskReduction.time_reduction",
)


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
        list(POINT_COUNT_AXIS.parameters(asv_pr_check_enabled())),
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
        list(DASK_CUTOFF_AXIS.parameters(asv_pr_check_enabled())),
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
