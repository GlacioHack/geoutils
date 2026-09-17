"""Measure pair sampling and variogram reduction separately, then through the complete public workflow."""

from __future__ import annotations

from typing import Any, Literal

from benchmarks.asv_suite import asv_parameter_values, asv_pr_check_enabled
from benchmarks.workflows.variography import (
    prepare_pair_pointcloud,
    prepare_pair_raster,
    prepare_variogram_pairs,
)
from geoutils._misc import import_optional
from geoutils.profiler import profile_call
from geoutils.stats import Variogram, variogram

############################
# Shared measurement setup #
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

    def track_peak_process_tree_mem_mb(self, *parameters: Any) -> float:
        """Measure process memory including prepared inputs and the completed result."""

        _, metrics = profile_call(self._execute, dask=False, include_children=True)
        assert metrics.peak_process_tree_mem_mb is not None
        return metrics.peak_process_tree_mem_mb

    def _execute(self) -> Any:
        """Compute the operation supplied by each concrete benchmark."""

        raise NotImplementedError


setattr(_VariographyBenchmark.track_peak_process_tree_mem_mb, "unit", "MB")


def _prepare_estimator(estimator: str) -> None:
    """Load and warm the optional estimator before timing its repeated application to distance bins."""

    # ASV records an unavailable optional package as a skipped case, and compilation stays outside timing
    try:
        import_optional("skgstat", package_name="scikit-gstat", extra_name="geostat")
    except ImportError as exc:
        raise NotImplementedError("Install geoutils[geostat] to measure variogram estimators") from exc
    Variogram.from_pairs(prepare_variogram_pairs(64), estimator=estimator, n_lags=4)


#########################################
# Reduction of already sampled pairs    #
#########################################


class VariogramPairCount(_VariographyBenchmark):
    """Vary pair count for the mean-square and robust median estimators at 24 fixed distance bins."""

    param_names = ["n_pairs", "estimator"]
    params = [asv_parameter_values([10_000, 100_000, 1_000_000], 1_000), ["matheron", "dowd"]]

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
    params = [asv_parameter_values([24, 100, 256], 24), ["matheron", "dowd"]]

    def setup(self, n_lags: int, estimator: str) -> None:
        """Keep the pair sample fixed while changing only its number of distance bins."""

        super().setup(1_000 if asv_pr_check_enabled() else 100_000, estimator)
        self.n_lags = n_lags


#########################################
# Spatial pair sampling                 #
#########################################


class RasterPairSampling(_VariographyBenchmark):
    """Compare all regular-grid sampling methods on prepared eager and Dask rasters."""

    param_names = ["n_pairs", "execution_mode", "sampling_method"]
    sampling_methods = ["independent", "anchors", "chunk_anchors", "anchor_batched", "random_xy"]
    params = [
        asv_parameter_values([1_000, 10_000, 100_000], 1_000),
        ["eager", "dask"],
        ["chunk_anchors", "random_xy"] if asv_pr_check_enabled() else sampling_methods,
    ]

    def setup(self, n_pairs: int, execution_mode: Literal["eager", "dask"], sampling_method: str) -> None:
        """Prepare one raster and fix batching, distance limits and the random seed for every method."""

        size = 256 if asv_pr_check_enabled() else 1024
        self._prepare(size, n_pairs, execution_mode, sampling_method)

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


class RasterPairSamplingSize(RasterPairSampling):
    """Vary source raster size around a fixed pair count and the default chunk-anchor strategy."""

    param_names = ["raster_size", "execution_mode"]
    params = [asv_parameter_values([256, 1024, 4096], 256), ["eager", "dask"]]

    def setup(self, raster_size: int, execution_mode: Literal["eager", "dask"]) -> None:
        """Keep 10,000 requested pairs while increasing the number of source chunks."""

        n_pairs = 1_000 if asv_pr_check_enabled() else 10_000
        self._prepare(raster_size, n_pairs, execution_mode, "chunk_anchors")


class PointPairSamplingSize(_VariographyBenchmark):
    """Compare exact ring searches and nearest-vector sampling as the irregular point set grows."""

    param_names = ["n_points", "strategy"]
    params = [asv_parameter_values([1_000, 10_000, 100_000], 1_000), ["kdtree", "hashgrid", "nn_logvector"]]

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
# Complete variogram workflow           #
#########################################


class RasterVariogramSize(RasterPairSamplingSize):
    """Measure public variogram() from pair generation through 24 robust distance-bin estimates."""

    def setup(self, raster_size: int, execution_mode: Literal["eager", "dask"]) -> None:
        """Prepare the raster and estimator while leaving sampling and reduction inside timing."""

        super().setup(raster_size, execution_mode)
        _prepare_estimator("dowd")

    def _execute(self) -> Variogram:
        """Sample and reduce pairs through the public stats module with one Dask thread."""

        with self.dask.config.set(scheduler="threads", num_workers=1):
            return variogram(self.source, estimator="dowd", n_lags=24, **self.pair_kwargs)
