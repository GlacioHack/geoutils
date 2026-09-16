"""Compare GeoUtils top-k sampling with Dask's native reduction."""

from __future__ import annotations

from typing import Literal

import numpy as np
from rasterio.transform import from_origin

import geoutils as gu
from benchmarks.asv_suite import asv_parameter_values
from geoutils._misc import import_optional
from geoutils.profiler import profile_call
from geoutils.sampling.subsampling import _splitmix64


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
        asv_parameter_values([256, 2048, 16384], pr_check_value=256),
    ]

    def setup(self, implementation: Literal["geoutils", "dask_argtopk"], subsample_size: int) -> None:
        """Prepare one lazy raster containing regular nodata cells."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        shape = (2048, 2048)
        rows = da.arange(shape[0], chunks=512)[:, None]
        columns = da.arange(shape[1], chunks=512)[None, :]
        positions = rows * shape[1] + columns
        values = da.where(positions % 19 == 0, np.nan, 1.0).astype(np.float32)
        self.raster = gu.RasterAccessor.from_array(values, from_origin(0, shape[0], 1, 1), 32633)

    def _run(self, implementation: Literal["geoutils", "dask_argtopk"], subsample_size: int) -> None:
        """Compute selected cell numbers through one top-k implementation."""

        import dask
        import dask.array as da

        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils":
                rows, columns = self.raster.rst.subsample(
                    subsample_size,
                    return_indices=True,
                    random_state=42,
                    strategy="topk",
                )
                dask.compute(rows, columns)
                return

            # Apply Dask's reduction to the same valid cells and SplitMix64 keys used by GeoUtils
            values = self.raster.data.reshape(-1)
            valid = da.isfinite(values)
            valid_count = int(valid.sum().compute())
            count = min(subsample_size, valid_count)
            cell_numbers = da.arange(values.size, chunks=values.chunks, dtype=np.int64)
            key_input = np.uint64(42) ^ cell_numbers.astype(np.uint64)
            keys = key_input.map_blocks(_splitmix64, dtype=np.uint64)
            eligible_keys = da.where(valid, keys, np.iinfo(np.uint64).max)
            da.argtopk(eligible_keys, -count, split_every=8).compute()

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
        asv_parameter_values([262_145, 524_288, 1_048_576], pr_check_value=262_145),
    ]

    def setup(self, implementation: Literal["geoutils_cutoff", "dask_topk"], subsample_size: int) -> None:
        """Prepare equivalent lazy keys and raster chunks for both selection methods."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        shape = (2048, 2048)
        rows = da.arange(shape[0], chunks=512)[:, None]
        columns = da.arange(shape[1], chunks=512)[None, :]
        positions = rows * shape[1] + columns
        self.values = da.where(positions % 19 == 0, np.nan, 1.0).astype(np.float32)
        self.shape = shape

        row_chunks, column_chunks = self.values.chunks
        row_starts = np.cumsum((0, *row_chunks))
        column_starts = np.cumsum((0, *column_chunks))
        self.tiles = np.array(
            [
                (row_starts[row], row_starts[row + 1], column_starts[column], column_starts[column + 1])
                for row in range(len(row_chunks))
                for column in range(len(column_chunks))
            ],
            dtype=np.int64,
        )
        self.blocks = self.values.to_delayed().ravel().tolist()
        self.largest_chunk = max(int(rows * columns) for rows in row_chunks for columns in column_chunks)

    def _run(self, implementation: Literal["geoutils_cutoff", "dask_topk"], subsample_size: int) -> None:
        """Find the exact selection boundary with GeoUtils or Dask's native reduction."""

        import dask
        import dask.array as da

        with dask.config.set(scheduler="threads", num_workers=1):
            if implementation == "geoutils_cutoff":
                from geoutils.interface.raster_point import _dask_raster_topk_cutoff

                sample_size, _, cutoff = _dask_raster_topk_cutoff(
                    self.blocks,
                    self.tiles,
                    self.shape,
                    self.largest_chunk,
                    subsample_size,
                    True,
                    42,
                )
                if sample_size != subsample_size or cutoff is None:
                    raise AssertionError("The cutoff search did not find the requested selection boundary.")
                return

            values = self.values.reshape(-1)
            valid = da.isfinite(values)
            cell_numbers = da.arange(values.size, chunks=values.chunks, dtype=np.int64)
            key_input = np.uint64(42) ^ cell_numbers.astype(np.uint64)
            keys = key_input.map_blocks(_splitmix64, dtype=np.uint64)
            eligible_keys = da.where(valid, keys, np.iinfo(np.uint64).max)
            da.topk(eligible_keys, -subsample_size, split_every=8).compute()

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
