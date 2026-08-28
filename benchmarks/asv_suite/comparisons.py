"""Measure parameter-dependent time and RAM across GeoUtils backends and GDAL."""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from typing import Literal, cast

from benchmarks.gdal_comparison.commands import ComparisonOperation
from benchmarks.gdal_comparison.runner import GdalRunner
from benchmarks.workflows.registry import OperationName
from benchmarks.workflows.runner import BenchmarkConfig, BenchmarkRunner
from geoutils.interface.gridding import GriddingEngine, GriddingMethod

Implementation = Literal["eager", "dask", "multiprocessing", "gdal"]


@dataclass(frozen=True)
class Comparison:
    """Describe one parameter plot and the ASV classes that supply its lines."""

    slug: str
    title: str
    parameter_label: str
    series: tuple[tuple[str, str], ...]
    logarithmic_x: bool = False
    documentation: bool = True


# The renderer uses these exact public class names to find each series in saved ASV results
COMPARISONS: tuple[Comparison, ...] = (
    Comparison(
        slug="interpolation-point-count",
        title="Interpolation point count",
        parameter_label="Interpolated points",
        series=(
            ("Eager", "EagerInterpolationPointCount"),
            ("Dask", "DaskInterpolationPointCount"),
            ("Multiprocessing", "MultiprocessingInterpolationPointCount"),
        ),
        logarithmic_x=True,
    ),
    Comparison(
        slug="reprojection-raster-size",
        title="Reprojection raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerReprojectionRasterSize"),
            ("Dask", "DaskReprojectionRasterSize"),
            ("Multiprocessing", "MultiprocessingReprojectionRasterSize"),
            ("GDAL", "GdalReprojectionRasterSize"),
        ),
    ),
    Comparison(
        slug="filter-chunk-size",
        title="Filter chunk size",
        parameter_label="Square chunk width and height (pixels)",
        series=(
            ("Dask", "DaskFilterChunkSize"),
            ("Multiprocessing", "MultiprocessingFilterChunkSize"),
        ),
    ),
    Comparison(
        slug="polygonization-raster-size",
        title="Polygonization raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerPolygonizationRasterSize"),
            ("Dask", "DaskPolygonizationRasterSize"),
            ("Multiprocessing", "MultiprocessingPolygonizationRasterSize"),
            ("GDAL", "GdalPolygonizationRasterSize"),
        ),
    ),
    Comparison(
        slug="rasterization-raster-size",
        title="Rasterization raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerRasterizationRasterSize"),
            ("Dask", "DaskRasterizationRasterSize"),
            ("Multiprocessing", "MultiprocessingRasterizationRasterSize"),
            ("GDAL", "GdalRasterizationRasterSize"),
        ),
    ),
    Comparison(
        slug="gridding-raster-size",
        title="Nearest gridding raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerGriddingRasterSize"),
            ("Dask", "DaskGriddingRasterSize"),
            ("Multiprocessing", "MultiprocessingGriddingRasterSize"),
            ("GDAL", "GdalGriddingRasterSize"),
        ),
    ),
    Comparison(
        slug="linear-gridding-raster-size",
        title="Linear gridding raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerLinearGriddingRasterSize"),
            ("Dask", "DaskLinearGriddingRasterSize"),
            ("Multiprocessing", "MultiprocessingLinearGriddingRasterSize"),
            ("GDAL", "GdalLinearGriddingRasterSize"),
        ),
        documentation=False,
    ),
    Comparison(
        slug="idw-gridding-raster-size",
        title="Inverse-distance gridding raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerIdwGriddingRasterSize"),
            ("Eager (Numba)", "EagerNumbaIdwGriddingRasterSize"),
            ("Dask", "DaskIdwGriddingRasterSize"),
            ("Multiprocessing", "MultiprocessingIdwGriddingRasterSize"),
            ("GDAL", "GdalIdwGriddingRasterSize"),
        ),
        documentation=False,
    ),
    Comparison(
        slug="mean-gridding-raster-size",
        title="Circular-mean gridding raster size",
        parameter_label="Square raster width and height (pixels)",
        series=(
            ("Eager", "EagerMeanGriddingRasterSize"),
            ("Eager (Numba)", "EagerNumbaMeanGriddingRasterSize"),
            ("Dask", "DaskMeanGriddingRasterSize"),
            ("Multiprocessing", "MultiprocessingMeanGriddingRasterSize"),
            ("GDAL", "GdalMeanGriddingRasterSize"),
        ),
        documentation=False,
    ),
    Comparison(
        slug="nearest-gridding-engine-point-count",
        title="Nearest gridding calculation engine",
        parameter_label="Source points per axis",
        series=(
            ("SciPy", "ScipyNearestGriddingPointCount"),
            ("Numba", "NumbaNearestGriddingPointCount"),
            ("GDAL", "GdalNearestGriddingPointCount"),
        ),
        documentation=False,
    ),
)


class _ImplementationBenchmark:
    """Share ASV settings, deterministic sources and complete result computation."""

    # A leading underscore keeps this shared class out of ASV's displayed benchmark list
    # Allow one complete operation to run for up to 15 minutes
    timeout = 900

    # Run once per timing sample, collect two samples and avoid an extra warm-up pass
    number = 1
    repeat = 2
    rounds = 1
    warmup_time = 0
    implementation: Implementation
    operation: OperationName

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Build the fixed configuration around one selected numeric parameter."""

        raise NotImplementedError

    def setup(self, parameter: int) -> None:
        """Prepare deterministic files and initialize one implementation."""

        # ASV calls setup before measuring, so input creation is excluded from every result
        # An explicit directory lets a fresh backend reuse sources for end-to-end timing
        self._tmpdir = tempfile.TemporaryDirectory(prefix="geoutils-asv-comparison-")
        self.config = self.make_config(parameter)
        self.config.directory = self._tmpdir.name
        self.sources = BenchmarkRunner("eager", self.config).prepare_sources()

        # GDAL needs only its prepared command while GeoUtils may need worker processes
        if self.implementation == "gdal":
            operation = cast(ComparisonOperation, self.operation)
            self.runner: BenchmarkRunner | GdalRunner = GdalRunner(operation, self.config, self.sources)
        else:
            self.runner = BenchmarkRunner(self.implementation, self.config).start()

    def teardown(self, parameter: int) -> None:
        """Stop workers and remove generated source, output and spill files."""

        # ASV calls teardown after each measurement to isolate parameters and implementations
        # Both runners leave the explicitly owned directory for this class to remove
        self.runner.close()
        if self.sources is not self.runner:
            self.sources.close()
        self._tmpdir.cleanup()

    def time_operation(self, parameter: int) -> None:
        """Measure a complete operation after implementation initialization."""

        # The time_ prefix tells ASV to time this method automatically
        # Every lazy result reaches a small value or a complete file before returning
        if isinstance(self.runner, GdalRunner):
            self.runner._execute()
        else:
            self.runner._execute(self.operation)

    def track_end_to_end_time_s(self, parameter: int) -> float:
        """Measure implementation initialization followed by one complete operation."""

        # The track_ prefix tells ASV to store the returned numeric measurement
        # GDAL has no persistent backend, so its command time is also its end-to-end time
        if self.implementation == "gdal":
            start_time = time.perf_counter()
            assert isinstance(self.runner, GdalRunner)
            self.runner._execute()
            return time.perf_counter() - start_time

        # Close the initialized backend while retaining the source files prepared by setup
        self.runner.close()
        fresh_runner = BenchmarkRunner(self.implementation, self.config)
        start_time = time.perf_counter()
        try:
            fresh_runner.start()
            fresh_runner._execute(self.operation)
        finally:
            elapsed_time_s = time.perf_counter() - start_time
            fresh_runner.close()
        self.runner = fresh_runner
        return elapsed_time_s

    def track_peak_process_tree_rss_mb(self, parameter: int) -> float:
        """Measure peak RAM for the benchmark process and implementation children."""

        # ASV stores this returned value separately from the two time measurements
        # A separate pass keeps memory sampling overhead out of elapsed-time measurements
        if isinstance(self.runner, GdalRunner):
            return self.runner.run().peak_process_tree_rss_mb
        return self.runner.run(self.operation).peak_process_tree_rss_mb


# ASV reads unit attributes from tracker functions when labelling stored results
setattr(_ImplementationBenchmark.track_end_to_end_time_s, "unit", "seconds")
setattr(_ImplementationBenchmark.track_peak_process_tree_rss_mb, "unit", "MB")


# Each private class defines one parameter axis while public subclasses select the implementation
class _InterpolationPointCount(_ImplementationBenchmark):
    """Keep raster and chunk sizes fixed while varying interpolated points."""

    operation: OperationName = "interp_points"

    # ASV uses this label in results and passes each listed value as parameter
    param_names = ["interpolated_points"]
    params = [[256, 2048, 16384]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected point count in an otherwise fixed configuration."""

        # The same raster and chunks isolate point-selection and interpolation work
        return BenchmarkConfig(shape=(2048, 2048), chunks=(512, 512), ninterp=parameter)


class EagerInterpolationPointCount(_InterpolationPointCount):
    """Measure eager interpolation as the requested point count grows."""

    implementation: Implementation = "eager"


class DaskInterpolationPointCount(_InterpolationPointCount):
    """Measure Dask interpolation as the requested point count grows."""

    implementation: Implementation = "dask"


class MultiprocessingInterpolationPointCount(_InterpolationPointCount):
    """Measure multiprocessing interpolation as the requested point count grows."""

    implementation: Implementation = "multiprocessing"


class _ReprojectionRasterSize(_ImplementationBenchmark):
    """Keep chunk size fixed while varying input and output raster size."""

    operation: OperationName = "reproject"

    # ASV uses this label in results and passes each listed value as parameter
    param_names = ["raster_size"]
    params = [[1024, 2048, 4096]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size in an otherwise fixed configuration."""

        # Fixed chunks expose the scaling cost of processing more raster pixels
        return BenchmarkConfig(shape=(parameter, parameter), chunks=(512, 512))


class EagerReprojectionRasterSize(_ReprojectionRasterSize):
    """Measure eager reprojection as input and output rasters grow."""

    implementation: Implementation = "eager"


class DaskReprojectionRasterSize(_ReprojectionRasterSize):
    """Measure Dask reprojection as input and output rasters grow."""

    implementation: Implementation = "dask"


class MultiprocessingReprojectionRasterSize(_ReprojectionRasterSize):
    """Measure multiprocessing reprojection as input and output rasters grow."""

    implementation: Implementation = "multiprocessing"


class GdalReprojectionRasterSize(_ReprojectionRasterSize):
    """Measure GDAL reprojection as input and output rasters grow."""

    implementation: Implementation = "gdal"


class _FilterChunkSize(_ImplementationBenchmark):
    """Keep raster size and filter window fixed while varying square chunks."""

    operation: OperationName = "filter"

    # ASV uses this label in results and passes each listed value as parameter
    param_names = ["chunk_size"]
    params = [[256, 512, 1024]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected chunk size in an otherwise fixed configuration."""

        # Fixed raster pixels isolate task scheduling and overlap granularity
        return BenchmarkConfig(shape=(2048, 2048), chunks=(parameter, parameter))


class DaskFilterChunkSize(_FilterChunkSize):
    """Measure Dask filtering as scheduler and overlap granularity change."""

    implementation: Implementation = "dask"


class MultiprocessingFilterChunkSize(_FilterChunkSize):
    """Measure multiprocessing filtering as task and overlap granularity change."""

    implementation: Implementation = "multiprocessing"


class _PolygonizationRasterSize(_ImplementationBenchmark):
    """Keep connected-region count fixed while varying the raster size."""

    operation: OperationName = "polygonize"

    # ASV uses this label in results and passes each listed value as parameter
    param_names = ["raster_size"]
    params = [[1024, 2048, 4096]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size around a fixed set of regions."""

        # A constant 441 regions isolates the cost of visiting more raster pixels
        return BenchmarkConfig(
            shape=(parameter, parameter),
            chunks=(512, 512),
            polygon_regions_per_axis=21,
        )


class EagerPolygonizationRasterSize(_PolygonizationRasterSize):
    """Measure eager polygonization as the source raster grows."""

    implementation: Implementation = "eager"


class DaskPolygonizationRasterSize(_PolygonizationRasterSize):
    """Measure Dask polygonization as the source raster grows."""

    implementation: Implementation = "dask"


class MultiprocessingPolygonizationRasterSize(_PolygonizationRasterSize):
    """Measure multiprocessing polygonization as the source raster grows."""

    implementation: Implementation = "multiprocessing"


class GdalPolygonizationRasterSize(_PolygonizationRasterSize):
    """Measure GDAL polygonization as the source raster grows."""

    implementation: Implementation = "gdal"


class _RasterizationRasterSize(_ImplementationBenchmark):
    """Keep vector complexity fixed while varying the output raster size."""

    operation: OperationName = "rasterize"

    # ASV uses this label in results and passes each listed value as parameter
    param_names = ["raster_size"]
    params = [[1024, 2048, 4096]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size around a fixed vector input."""

        # A constant 2,601 polygons isolates the cost of producing more output pixels
        return BenchmarkConfig(
            shape=(parameter, parameter),
            chunks=(512, 512),
            vector_features_per_axis=51,
        )


class EagerRasterizationRasterSize(_RasterizationRasterSize):
    """Measure eager rasterization as the output raster grows."""

    implementation: Implementation = "eager"


class DaskRasterizationRasterSize(_RasterizationRasterSize):
    """Measure Dask rasterization as the output raster grows."""

    implementation: Implementation = "dask"


class MultiprocessingRasterizationRasterSize(_RasterizationRasterSize):
    """Measure multiprocessing rasterization as the output raster grows."""

    implementation: Implementation = "multiprocessing"


class GdalRasterizationRasterSize(_RasterizationRasterSize):
    """Measure GDAL rasterization as the output raster grows."""

    implementation: Implementation = "gdal"


class _GriddingRasterSize(_ImplementationBenchmark):
    """Keep one gridding method and point count fixed while varying raster size."""

    operation: OperationName = "grid"
    grid_resampling: GriddingMethod = "nearest"
    grid_engine: GriddingEngine = "scipy"
    grid_dist_nodata_pixel = float("inf")
    point_features_per_axis = 3

    # ASV uses this label in results and passes each listed value as parameter
    param_names = ["raster_size"]
    params = [[512, 1024, 2048]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected raster size around a fixed point-cloud input."""

        # Method-specific inputs remain fixed while only the number of output cells changes
        return BenchmarkConfig(
            shape=(parameter, parameter),
            chunks=(512, 512),
            point_features_per_axis=self.point_features_per_axis,
            grid_resampling=self.grid_resampling,
            grid_dist_nodata_pixel=self.grid_dist_nodata_pixel,
            grid_engine=self.grid_engine,
        )


class EagerGriddingRasterSize(_GriddingRasterSize):
    """Measure eager gridding as the output raster grows."""

    implementation: Implementation = "eager"


class DaskGriddingRasterSize(_GriddingRasterSize):
    """Measure Dask gridding as the output raster grows."""

    implementation: Implementation = "dask"


class MultiprocessingGriddingRasterSize(_GriddingRasterSize):
    """Measure multiprocessing gridding as the output raster grows."""

    implementation: Implementation = "multiprocessing"


class GdalGriddingRasterSize(_GriddingRasterSize):
    """Measure GDAL nearest gridding as the output raster grows."""

    implementation: Implementation = "gdal"


class _LinearGriddingRasterSize(_GriddingRasterSize):
    """Keep linear interpolation and source points fixed while varying raster size."""

    grid_resampling: GriddingMethod = "linear"
    point_features_per_axis = 9


class EagerLinearGriddingRasterSize(_LinearGriddingRasterSize):
    """Measure eager linear gridding as the output raster grows."""

    implementation: Implementation = "eager"


class DaskLinearGriddingRasterSize(_LinearGriddingRasterSize):
    """Measure Dask linear gridding as the output raster grows."""

    implementation: Implementation = "dask"


class MultiprocessingLinearGriddingRasterSize(_LinearGriddingRasterSize):
    """Measure multiprocessing linear gridding as the output raster grows."""

    implementation: Implementation = "multiprocessing"


class GdalLinearGriddingRasterSize(_LinearGriddingRasterSize):
    """Measure GDAL linear gridding as the output raster grows."""

    implementation: Implementation = "gdal"


class _IdwGriddingRasterSize(_GriddingRasterSize):
    """Keep inverse-distance support and source points fixed while varying raster size."""

    grid_resampling: GriddingMethod = "idw"
    grid_dist_nodata_pixel = 16.0
    point_features_per_axis = 17


class EagerIdwGriddingRasterSize(_IdwGriddingRasterSize):
    """Measure eager inverse-distance gridding as the output raster grows."""

    implementation: Implementation = "eager"


class EagerNumbaIdwGriddingRasterSize(_IdwGriddingRasterSize):
    """Measure eager Numba inverse-distance gridding as the output raster grows."""

    implementation: Implementation = "eager"
    grid_engine: GriddingEngine = "numba"


class DaskIdwGriddingRasterSize(_IdwGriddingRasterSize):
    """Measure Dask inverse-distance gridding as the output raster grows."""

    implementation: Implementation = "dask"


class MultiprocessingIdwGriddingRasterSize(_IdwGriddingRasterSize):
    """Measure multiprocessing inverse-distance gridding as the output raster grows."""

    implementation: Implementation = "multiprocessing"


class GdalIdwGriddingRasterSize(_IdwGriddingRasterSize):
    """Measure GDAL inverse-distance gridding as the output raster grows."""

    implementation: Implementation = "gdal"


class _MeanGriddingRasterSize(_GriddingRasterSize):
    """Keep circular-mean support and source points fixed while varying raster size."""

    grid_resampling: GriddingMethod = "mean"
    grid_dist_nodata_pixel = 16.0
    point_features_per_axis = 17


class EagerMeanGriddingRasterSize(_MeanGriddingRasterSize):
    """Measure eager circular-mean gridding as the output raster grows."""

    implementation: Implementation = "eager"


class EagerNumbaMeanGriddingRasterSize(_MeanGriddingRasterSize):
    """Measure eager Numba circular-mean gridding as the output raster grows."""

    implementation: Implementation = "eager"
    grid_engine: GriddingEngine = "numba"


class DaskMeanGriddingRasterSize(_MeanGriddingRasterSize):
    """Measure Dask circular-mean gridding as the output raster grows."""

    implementation: Implementation = "dask"


class MultiprocessingMeanGriddingRasterSize(_MeanGriddingRasterSize):
    """Measure multiprocessing circular-mean gridding as the output raster grows."""

    implementation: Implementation = "multiprocessing"


class GdalMeanGriddingRasterSize(_MeanGriddingRasterSize):
    """Measure GDAL circular-mean gridding as the output raster grows."""

    implementation: Implementation = "gdal"


class _NearestGriddingPointCount(_ImplementationBenchmark):
    """Keep raster size fixed while varying source points for nearest gridding."""

    operation: OperationName = "grid"
    grid_engine: GriddingEngine = "scipy"

    # Points per axis gives nine, eighty-one and one thousand eighty-nine source points
    param_names = ["points_per_axis"]
    params = [[3, 9, 33]]

    def make_config(self, parameter: int) -> BenchmarkConfig:
        """Place the selected point count in an otherwise fixed configuration."""

        return BenchmarkConfig(
            shape=(1024, 1024),
            chunks=(512, 512),
            point_features_per_axis=parameter,
            grid_resampling="nearest",
            grid_dist_nodata_pixel=float("inf"),
            grid_engine=self.grid_engine,
        )


class ScipyNearestGriddingPointCount(_NearestGriddingPointCount):
    """Measure the SciPy nearest engine as source point count grows."""

    implementation: Implementation = "eager"


class NumbaNearestGriddingPointCount(_NearestGriddingPointCount):
    """Measure the Numba nearest engine as source point count grows."""

    implementation: Implementation = "eager"
    grid_engine: GriddingEngine = "numba"


class GdalNearestGriddingPointCount(_NearestGriddingPointCount):
    """Measure GDAL nearest gridding as source point count grows."""

    implementation: Implementation = "gdal"
