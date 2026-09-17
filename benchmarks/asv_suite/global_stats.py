"""Compare GeoUtils global Dask reduction with equivalent native Dask statistics."""

from __future__ import annotations

from typing import Literal

import numpy as np

from benchmarks.asv_suite import asv_parameter_values
from geoutils._misc import import_optional
from geoutils.stats.reduction import (
    _normalize_statistics,
    _reduce_values,
    _statistics_dask,
)


class GlobalDaskReduction:
    """Measure the shared GeoUtils reducer and native Dask reductions on the same prepared array."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    timeout = 300
    param_names = ["implementation", "raster_size"]
    params = [["geoutils", "native_dask"], asv_parameter_values([256, 1024, 4096], 256)]

    def setup(self, implementation: Literal["geoutils", "native_dask"], raster_size: int) -> None:
        """Prepare one finite Dask raster and the same mergeable statistic request for both implementations."""

        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        del implementation
        generator = np.random.default_rng(42)
        values = generator.normal(size=(raster_size, raster_size))
        values[::97, ::89] = np.nan
        self.values = da.from_array(values, chunks=(256, 256))
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
                estimates, count = _statistics_dask(self.values, self.aliases)
                dask.compute(estimates, count)
