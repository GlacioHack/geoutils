"""Define public variogram benchmarks and their deterministic inputs."""

from __future__ import annotations

from functools import wraps
from typing import Any, Literal

import numpy as np
import xarray as xr
from rasterio.transform import from_origin

import geoutils as gu
from benchmarks.asv_suite import asv_pr_check_enabled
from benchmarks.workflows.config import (
    POINT_COUNT_AXIS,
    RASTER_AXIS,
    VARIOGRAM_LAG_AXIS,
    VARIOGRAM_PAIR_AXIS,
    process_tree_memory_increase_mb,
)
from geoutils._misc import import_optional
from geoutils.profiler import profile_call
from geoutils.stats import Variogram, variogram


def prepare_variogram_pairs(n_pairs: int) -> xr.Dataset:
    """Create complete endpoint values at log-uniform distances, independently of spatial pair sampling.

    A fixed seed gives every estimator identical float64 inputs. The increasing difference amplitude creates a
    nonconstant variogram, while random endpoint offsets avoid a special case with one constant endpoint.
    """

    # Spread observations across short and long distances without enumerating a spatial distance matrix
    rng = np.random.default_rng(42)
    distances = np.exp(rng.uniform(0, np.log(1024), n_pairs))
    first = rng.normal(size=n_pairs)
    differences = rng.normal(size=n_pairs) * np.sqrt(1 - np.exp(-distances / 100))

    # Keep only the public pair layout consumed by Variogram.from_pairs()
    return xr.Dataset(
        {
            "distance": ("pair", distances),
            "value": (("pair", "endpoint"), np.column_stack((first, first + differences))),
        },
        attrs={"min_distance": 1.0, "max_distance": 1024.0},
    )


def prepare_pair_raster(size: int, execution_mode: Literal["eager", "dask"]) -> Any:
    """Create a smooth projected raster with scattered missing cells and 256 by 256 Dask chunks.

    Both modes start from the same prepared float32 values. Dask measures selected chunk reads and task scheduling
    from memory; this fixture does not measure disk throughput or claim a larger-than-memory contract.
    """

    # Vary values in both directions and remove a known fraction of cells to require finite endpoint checks
    rows, columns = np.arange(size)[:, None], np.arange(size)[None, :]
    values = (np.sin(columns / 31) + np.cos(rows / 53)).astype(np.float32)
    values[(rows * size + columns) % 17 == 0] = np.nan
    transform = from_origin(0, size, 1, 1)

    # Expose the public object method in both modes while leaving lazy values uncomputed
    if execution_mode == "dask":
        import_optional("dask", extra_name="benchmark")
        import dask.array as da

        array = da.from_array(values, chunks=(256, 256))
        return gu.RasterAccessor.from_array(array, transform, 32633, nodata=-99999).rst
    return gu.Raster.from_array(values, transform, 32633, nodata=-99999)


def prepare_pair_pointcloud(n_points: int) -> gu.PointCloud:
    """Create irregular projected points at roughly unit spacing with finite, smoothly varying values."""

    # Keep average point density constant so increasing point count increases the search extent
    rng = np.random.default_rng(42)
    coordinates = rng.uniform(0, np.sqrt(n_points), size=(n_points, 2))
    x, y = coordinates.T
    values = np.sin(x / 31) + np.cos(y / 53)
    return gu.PointCloud.from_xyz(x, y, values, crs=32633)


############################
# Public variogram workflow
############################


def _prepare_estimator(estimator: str) -> None:
    """Load and warm the optional estimator before timing variogram()."""

    # ASV records an unavailable optional package as a skipped case, and compilation stays outside timing
    try:
        import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
    except ImportError as exc:
        raise NotImplementedError("Install geoutils[geostat] to measure variogram estimators") from exc
    Variogram.from_pairs(prepare_variogram_pairs(64), estimator=estimator, n_lags=4)


class RasterVariogramSize:
    """Measure public variogram() while varying raster size and eager or Dask execution."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    timeout = 300
    param_names = ["raster_size", "execution_mode"]
    params = [list(RASTER_AXIS.parameters(asv_pr_check_enabled())), ["eager", "dask"]]

    def setup(self, raster_size: int, execution_mode: Literal["eager", "dask"]) -> None:
        """Prepare the raster and estimator while leaving sampling and reduction inside timing."""

        # Keep the same values and sampling request while changing the source size and loading mode
        self.source = prepare_pair_raster(raster_size, execution_mode)
        self.dask = import_optional("dask", extra_name="benchmark")
        n_pairs = 1_000 if asv_pr_check_enabled() else 10_000
        self.pair_kwargs: dict[str, Any] = {
            "n_pairs": n_pairs,
            "sampling": "loglag",
            "strategy": "chunk_anchors",
            "min_distance": 1,
            "max_distance": raster_size / 2,
            "batch_pairs": 100_000,
            "anchors_per_round": 2_000,
            "random_state": 42,
        }
        _prepare_estimator("dowd")

    def _execute(self) -> Variogram:
        """Sample and reduce pairs through public variogram() with one Dask thread."""

        with self.dask.config.set(scheduler="threads", num_workers=1):
            return variogram(self.source, estimator="dowd", n_lags=24, **self.pair_kwargs)

    def time_operation(self, raster_size: int, execution_mode: Literal["eager", "dask"]) -> None:
        """Compute the complete variogram from the prepared raster."""

        self._execute()

    def track_process_tree_mem_increase_mb(self, raster_size: int, execution_mode: Literal["eager", "dask"]) -> float:
        """Measure peak memory increase while computing the complete variogram."""

        _, metrics = profile_call(self._execute, dask=False, include_children=True)
        return process_tree_memory_increase_mb(metrics)


# Keep the stored ASV history while locating the benchmark beside its public operation
setattr(
    RasterVariogramSize.time_operation,
    "benchmark_name",
    "asv_suite.variography.RasterVariogramSize.time_operation",
)
setattr(
    RasterVariogramSize.track_process_tree_mem_increase_mb,
    "benchmark_name",
    "asv_suite.variography.RasterVariogramSize.track_process_tree_mem_increase_mb",
)
setattr(RasterVariogramSize.track_process_tree_mem_increase_mb, "unit", "MB")
from geoutils.stats import Variogram

############################
# Shared measurement setup
############################


class _VariographyBenchmark:
    """Measure complete results after constructing inputs and importing optional estimators."""

    number = 1
    repeat = 3
    rounds = 1
    warmup_time = 0
    timeout = 300

    def time_operation(self, *parameters: Any) -> None:
        """Compute the complete pair dataset or reduced variogram from prepared inputs."""

        self._execute()

    def track_process_tree_mem_increase_mb(self, *parameters: Any) -> float:
        """Measure peak memory increase while producing the completed result."""

        _, metrics = profile_call(self._execute, dask=False, include_children=True)
        return process_tree_memory_increase_mb(metrics)

    def _execute(self) -> Any:
        """Compute the operation supplied by each concrete benchmark."""

        raise NotImplementedError


setattr(_VariographyBenchmark.track_process_tree_mem_increase_mb, "unit", "MB")


#########################################
# Reduction of already sampled pairs
#########################################


class VariogramPairCount(_VariographyBenchmark):
    """Vary pair count for the mean-square and robust median estimators at 24 fixed distance bins."""

    param_names = ["n_pairs", "estimator"]
    params = [list(VARIOGRAM_PAIR_AXIS.parameters(asv_pr_check_enabled())), ["matheron", "dowd"]]

    def setup(self, n_pairs: int, estimator: str) -> None:
        """Prepare the same finite pairs for both estimators outside the measured call."""

        _prepare_estimator(estimator)
        self.pairs = prepare_variogram_pairs(n_pairs)
        self.estimator = estimator
        self.n_lags = 24

    def _execute(self) -> Variogram:
        """Reduce all prepared pairs to their distance-bin estimates and counts."""

        return Variogram.from_pairs(self.pairs, estimator=self.estimator, n_lags=self.n_lags)


class VariogramLagCount(VariogramPairCount):
    """Vary distance-bin count at 100,000 pairs to expose repeated full-input scans."""

    param_names = ["n_lags", "estimator"]
    params = [list(VARIOGRAM_LAG_AXIS.parameters(asv_pr_check_enabled())), ["matheron", "dowd"]]

    def setup(self, n_lags: int, estimator: str) -> None:
        """Keep the pair sample fixed while changing only its number of distance bins."""

        super().setup(1_000 if asv_pr_check_enabled() else 100_000, estimator)
        self.n_lags = n_lags


#########################################
# Spatial pair sampling
#########################################


class _RasterPairSamplingBenchmark(_VariographyBenchmark):
    """Share prepared raster sampling and execution between the two component benchmarks."""

    def _prepare(self, size: int, n_pairs: int, execution_mode: Literal["eager", "dask"], sampling_method: str) -> None:
        """Share the source and sampling settings between pair-count and raster-size comparisons."""

        # Keep the same source values and worker count for every sampling method
        self.source = prepare_pair_raster(size, execution_mode)
        self.dask = import_optional("dask", extra_name="benchmark")

        # Bound candidate batches and reuse the same map-distance range and random seed
        self.pair_kwargs = {
            "n_pairs": n_pairs,
            "sampling": "random_xy" if sampling_method == "random_xy" else "loglag",
            "strategy": "chunk_anchors" if sampling_method == "random_xy" else sampling_method,
            "min_distance": 1,
            "max_distance": size / 2,
            "batch_pairs": 100_000,
            "anchors_per_round": 2_000,
            "random_state": 42,
        }

    def _execute(self) -> Any:
        """Draw finite pairs and construct all endpoint values and coordinates with one Dask thread."""

        with self.dask.config.set(scheduler="threads", num_workers=1):
            return self.source.pairsample(**self.pair_kwargs)


class RasterPairSampling(_RasterPairSamplingBenchmark):
    """Compare all regular-grid sampling methods on prepared eager and Dask rasters."""

    param_names = ["n_pairs", "execution_mode", "sampling_method"]
    sampling_methods = ["independent", "anchors", "chunk_anchors", "anchor_batched", "random_xy"]
    params = [
        list(POINT_COUNT_AXIS.parameters(asv_pr_check_enabled())),
        ["eager", "dask"],
        ["chunk_anchors", "random_xy"] if asv_pr_check_enabled() else sampling_methods,
    ]

    def setup(self, n_pairs: int, execution_mode: Literal["eager", "dask"], sampling_method: str) -> None:
        """Prepare one raster and fix batching, distance limits and the random seed for every method."""

        size = 256 if asv_pr_check_enabled() else 1024
        self._prepare(size, n_pairs, execution_mode, sampling_method)


class RasterPairSamplingSize(_RasterPairSamplingBenchmark):
    """Vary source raster size around a fixed pair count and the default chunk-anchor strategy."""

    param_names = ["raster_size", "execution_mode"]
    params = [list(RASTER_AXIS.parameters(asv_pr_check_enabled())), ["eager", "dask"]]

    def setup(self, raster_size: int, execution_mode: Literal["eager", "dask"]) -> None:
        """Keep 10,000 requested pairs while increasing the number of source chunks."""

        n_pairs = 1_000 if asv_pr_check_enabled() else 10_000
        self._prepare(raster_size, n_pairs, execution_mode, "chunk_anchors")


class PointPairSamplingSize(_VariographyBenchmark):
    """Compare exact ring searches and nearest-vector sampling as the irregular point set grows."""

    param_names = ["n_points", "strategy"]
    params = [
        list(POINT_COUNT_AXIS.parameters(asv_pr_check_enabled())),
        ["kdtree", "hashgrid", "nn_logvector"],
    ]

    def setup(self, n_points: int, strategy: str) -> None:
        """Prepare a constant-density point cloud and enough nearby candidates for each search method."""

        self.source = prepare_pair_pointcloud(n_points)
        self.pair_kwargs = {
            "n_pairs": 200 if asv_pr_check_enabled() else 2_000,
            "strategy": strategy,
            "min_distance": 1,
            "max_distance": n_points**0.5 / 2,
            "anchors_per_round": 2_000,
            "nn_tolerance": 0.5,
            "random_state": 42,
        }

    def _execute(self) -> Any:
        """Build the spatial search, sample finite pairs and construct their complete labelled dataset."""

        return self.source.pairsample(**self.pair_kwargs)


#########################################
# Stable internal benchmark identifiers
#########################################


def _named_method(method: Any, benchmark_name: str) -> Any:
    """Copy one inherited measurement method with its existing ASV identifier."""

    @wraps(method)
    def measured(self: Any, *parameters: Any) -> Any:
        return method(self, *parameters)

    setattr(measured, "benchmark_name", benchmark_name)
    return measured


# Keep existing ASV history while locating component measurements outside the public operation modules
for _benchmark_class in (
    VariogramPairCount,
    VariogramLagCount,
    RasterPairSampling,
    RasterPairSamplingSize,
    PointPairSamplingSize,
):
    for _method_name in ("time_operation", "track_process_tree_mem_increase_mb"):
        _method = getattr(_benchmark_class, _method_name)
        _benchmark_name = f"asv_suite.variography.{_benchmark_class.__name__}.{_method_name}"
        setattr(_benchmark_class, _method_name, _named_method(_method, _benchmark_name))
