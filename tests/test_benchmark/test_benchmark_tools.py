"""Minimal tests for benchmark workflows, reports and external CLI commands."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import rasterio as rio
import xarray as xr

from benchmarks.asv_suite import parameter_sweeps as benchmark_parameter_sweeps
from benchmarks.asv_suite.parameter_sweeps import (
    BENCHMARK_CASE_BY_CLASS,
    BENCHMARK_CASES,
    COMPARISONS,
    EXTERNAL_REFERENCE_CASE_BY_CLASS,
)
from benchmarks.asv_suite.render_results import (
    COMPARISON_REPORT_DIRECTORY,
    DOCUMENTATION_DATA,
    DOCUMENTATION_MEMORY_PLOT,
    DOCUMENTATION_MEMORY_SCALING_PLOT,
    DOCUMENTATION_PDAL_PLOT,
    DOCUMENTATION_TIME_PLOT,
    PERFORMANCE_CHANGE_REPORT,
    SCALING_REPORT_PAGE,
    _PreviewResult,
    _render_preview,
    _select_complete_result,
    render_documentation_snapshot,
)
from benchmarks.gdal_comparison.commands import (
    COMPARISON_OPERATIONS,
    _warp_memory_limit_mb,
    build_gdal_command,
)
from benchmarks.pdal_comparison.commands import (
    PDAL_COMPARISON_OPERATIONS,
    build_pdal_command,
)
from benchmarks.workflows.grouped_reference import (
    compute_grouped_reference,
    prepare_grouped_reference,
)
from benchmarks.workflows.runner import BenchmarkConfig, BenchmarkRunner
from benchmarks.workflows.variography import (
    prepare_pair_pointcloud,
    prepare_pair_raster,
    prepare_variogram_pairs,
)
from geoutils import Variogram, _misc


class TestComparisonReport:
    """
    Test module for benchmark registration and report generation.

    Those are minimal tests to ensure changes to ``benchmarks/`` don't break the routines.
    Even if ASV runs quick checks of the benchmarking setup through CI, it's easier to have a detailed traceback here
    through Pytest for some aspects.
    We especially tests our custom routines/rendering for the benchmark webpage, comparisons to external refs (e.g.
    GDAL CLI), and across variables (raster/point size, chunk size, etc) and categories (e.g. eager/Dask/MP, method,
    etc) of interest.
    """

    def test_benchmark_registry__generated_classes_exist(self) -> None:
        """Checks that every registered benchmark and plotted series has a generated ASV class."""

        # Gather the GeoUtils/external classes registered for ASV and the classes referenced by report plots
        registered = set(BENCHMARK_CASE_BY_CLASS) | set(EXTERNAL_REFERENCE_CASE_BY_CLASS)
        plotted = {class_name for comparison in COMPARISONS for _, class_name in comparison.series}

        # Importing parameter_sweeps.py should create every class needed by ASV and the report
        assert BENCHMARK_CASES and COMPARISONS
        assert plotted <= registered
        assert all(hasattr(benchmark_parameter_sweeps, class_name) for class_name in registered)

    def test_render_preview__essential_files(self, tmp_path: Path) -> None:
        """Checks that preview rendering writes the main pages, data exports and plots."""

        pytest.importorskip("matplotlib")

        # Render the complete preview from the small fake ASV result bundled with the renderer
        _render_preview(tmp_path)
        report_directory = tmp_path / COMPARISON_REPORT_DIRECTORY

        # Keep this list to files used directly by the published benchmark site or downloaded result data
        expected_files = (
            tmp_path / "index.html",
            report_directory / "index.html",
            report_directory / SCALING_REPORT_PAGE,
            report_directory / "comparisons.csv",
            report_directory / "comparisons.json",
            report_directory / PERFORMANCE_CHANGE_REPORT,
        )
        assert all(path.is_file() for path in expected_files)
        assert any(report_directory.glob("*.svg"))

        # The JSON export must contain measurements rather than only an empty report shell
        payload = json.loads((report_directory / "comparisons.json").read_text(encoding="utf-8"))
        assert payload["measurements"]

    def test_select_complete_result__skips_incomplete_latest_run(self) -> None:
        """Checks that report selection falls back when the latest ASV result is incomplete."""

        # Make the second result newer, then remove one measurement that the report needs
        complete = _PreviewResult()
        complete.started_at = {"benchmark": 1}
        incomplete = _PreviewResult()
        incomplete.started_at = {"benchmark": 2}
        incomplete.values.pop(next(iter(incomplete.values)))

        # The earlier complete result should still be usable for report generation
        assert _select_complete_result((complete, incomplete)) is complete

    def test_render_documentation_snapshot__essential_files(self, tmp_path: Path) -> None:
        """Checks that documentation rendering writes its summary plots and numeric data."""

        pytest.importorskip("matplotlib")

        # Render the documentation files from the same small result used by preview mode
        records = render_documentation_snapshot(_PreviewResult(), tmp_path)

        # All graphics and their JSON source are needed when benchmark results are updated in the docs
        assert records
        assert (tmp_path / DOCUMENTATION_TIME_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_SCALING_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_PDAL_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_DATA).is_file()


class TestBenchmarkScenarios:
    """Test module for full scheduled inputs and reduced pull-request benchmark configurations."""

    @pytest.mark.parametrize(
        ("comparison_group", "parameters", "expected_shape", "expected_chunks"),
        (
            ("clip-raster-size", [2048, 4096, 8192], (8192, 8192), (2048, 2048)),
            ("reprojection-raster-size", [2048, 4096, 8192], (8192, 8192), (2048, 2048)),
            ("polygonization-raster-size", [2048, 4096, 8192], (8192, 8192), (2048, 2048)),
            ("rasterization-raster-size", [2048, 4096, 8192], (8192, 8192), (2048, 2048)),
            ("subsample-size", [16_384, 262_144, 1_048_576], (2048, 2048), (1024, 1024)),
            ("to-pointcloud-raster-size", [512, 1024, 2048], (2048, 2048), (1024, 1024)),
            ("gridding-raster-size", [2048, 4096, 8192], (8192, 8192), (2048, 2048)),
        ),
    )
    def test_external_comparison__scheduled_workloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        comparison_group: str,
        parameters: list[int],
        expected_shape: tuple[int, int],
        expected_chunks: tuple[int, int],
    ) -> None:
        """Checks that scheduled GDAL and PDAL comparisons use large inputs split into few large chunks."""

        # Select one generated GeoUtils class for the comparison and force the full scheduled profile
        monkeypatch.delenv("GEOUTILS_ASV_PR_CHECK", raising=False)
        case = next(case for case in BENCHMARK_CASES if case.comparison_group == comparison_group)
        benchmark = getattr(benchmark_parameter_sweeps, case.benchmark_class)()
        benchmark.operation_method = case.method

        # Build the largest configured workload and check both its input axis and bounded chunk layout
        config = benchmark.make_config(parameters[-1])
        assert benchmark.params == [parameters]
        assert config.shape == expected_shape
        assert config.chunks == expected_chunks

    @pytest.mark.parametrize(
        ("comparison_group", "parameter", "expected_shape", "expected_chunks"),
        (
            ("clip-raster-size", 1024, (1024, 1024), (1024, 1024)),
            ("reprojection-raster-size", 1024, (1024, 1024), (1024, 1024)),
            ("polygonization-raster-size", 1024, (1024, 1024), (1024, 1024)),
            ("rasterization-raster-size", 1024, (1024, 1024), (1024, 1024)),
            ("subsample-size", 256, (512, 512), (256, 256)),
            ("to-pointcloud-raster-size", 256, (256, 256), (256, 256)),
            ("gridding-raster-size", 512, (512, 512), (512, 512)),
        ),
    )
    def test_external_comparison__pull_request_workloads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        comparison_group: str,
        parameter: int,
        expected_shape: tuple[int, int],
        expected_chunks: tuple[int, int],
    ) -> None:
        """Checks that pull-request GDAL and PDAL cases keep small inputs for fast smoke testing."""

        # Select the same generated classes while enabling the lightweight pull-request configuration
        monkeypatch.setenv("GEOUTILS_ASV_PR_CHECK", "1")
        case = next(case for case in BENCHMARK_CASES if case.comparison_group == comparison_group)
        benchmark = getattr(benchmark_parameter_sweeps, case.benchmark_class)()
        benchmark.operation_method = case.method

        # The single pull-request parameter should build a much smaller source and execution chunk
        config = benchmark.make_config(parameter)
        assert config.shape == expected_shape
        assert config.chunks == expected_chunks


class TestBenchmarkProcess:
    """Test module for cache configuration applied inside benchmark worker processes."""

    def test_prepare_benchmark_process__gdal_cache_units(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that the worker config converts the benchmark's MiB cache size to Rasterio's byte value."""

        # Capture the live GDAL setting without changing the cache used by the test process
        configured_values: list[tuple[str, int]] = []
        monkeypatch.setattr(rio.env, "set_gdal_config", lambda name, value: configured_values.append((name, value)))
        monkeypatch.setattr(_misc, "_trim_process_memory", lambda: None)

        # Configure the same 64 MiB cache used by the benchmark workers
        _misc._prepare_benchmark_process(64)

        # Rasterio receives integer cache sizes in bytes rather than the MiB string accepted by GDAL commands
        assert configured_values == [("GDAL_CACHEMAX", 64 * 1024**2)]


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestBenchmarkRunner:
    """Test module for bounded operation outputs produced by BenchmarkRunner."""

    @pytest.mark.parametrize("execution_mode", ["dask", "multiprocessing"])
    def test_to_pointcloud__bounded_sample(
        self, execution_mode: Literal["dask", "multiprocessing"], tmp_path: Path
    ) -> None:
        """Checks that the large-data point conversion can request a bounded sample from either backend."""

        if execution_mode == "dask":
            pytest.importorskip("distributed")

        # Request more point rows than one 3 x 4 raster chunk to use the bounded cutoff path
        config = BenchmarkConfig(
            shape=(8, 10),
            chunks=(3, 4),
            pointcloud_subsample_size=17,
            directory=str(tmp_path / execution_mode),
        )

        # Run the shared workflow and let its internal count check validate all 17 output rows
        with BenchmarkRunner(execution_mode, config) as runner:
            result = runner.run("to_pointcloud", profile=False)

        # The constant raster gives the same compact correctness value through both backends
        assert result.value == config.raster_value

    @pytest.mark.parametrize("execution_mode", ["eager", "dask", "multiprocessing"])
    def test_clip__masks_outside_fixture_polygons(
        self,
        execution_mode: Literal["eager", "dask", "multiprocessing"],
        tmp_path: Path,
    ) -> None:
        """Checks that the clipping benchmark keeps polygon interiors and masks the surrounding raster."""

        if execution_mode == "dask":
            pytest.importorskip("distributed")

        # Use several polygons across multiple chunks so worker modes process inside and outside cells
        config = BenchmarkConfig(
            shape=(64, 64),
            chunks=(32, 32),
            vector_features_per_axis=3,
            directory=str(tmp_path / execution_mode),
        )

        # Run the complete benchmark workflow and inspect one kept center cell and one clipped corner
        with BenchmarkRunner(execution_mode, config) as runner:
            result = runner.run("clip", profile=False)
        assert result.output_file is not None
        with rio.open(result.output_file) as dataset:
            values = dataset.read(1)
            nodata = dataset.nodata

        # The odd polygon grid covers the raster center, while every fixture polygon stays away from the corners
        assert result.value == config.raster_value
        assert values[values.shape[0] // 2, values.shape[1] // 2] == config.raster_value
        assert not np.isfinite(values[0, 0]) or values[0, 0] == nodata


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestGroupedReferenceChunked:
    """Test module for running grouped benchmark workflows with each execution path."""

    @pytest.mark.parametrize("implementation", ["geoutils", "flox"])
    @pytest.mark.parametrize("execution_mode", ["eager", "dask"])
    def test_grouped_reference__runs(
        self,
        implementation: Literal["geoutils", "flox"],
        execution_mode: Literal["eager", "dask"],
    ) -> None:
        """Checks that GeoUtils and Flox return a complete table from eager and Dask inputs."""

        # Flox is an optional benchmark reference, so leave its two cases out of the base test environment
        if implementation == "flox":
            pytest.importorskip("flox")

        # Prepare four groups from small arrays, then run the same entry point used by the benchmark classes
        inputs = prepare_grouped_reference(16, 2, "interleaved", execution_mode)
        result = compute_grouped_reference(*inputs, implementation=implementation)

        # Four groups x two value arrays each contain count, mean and standard deviation
        assert result.shape == (4, 6)
        assert np.array_equal(result.index, np.arange(4))
        assert np.isfinite(result.to_numpy()).all()

    def test_grouped_reference__multiprocessing_benchmark_runs(self) -> None:
        """Checks that one generated multiprocessing benchmark can set up, run and clean up."""

        # Find the generated GeoUtils case used beside the Flox raster-size comparison
        case = next(
            case
            for case in BENCHMARK_CASES
            if case.comparison_group == "grouped-flox-raster-size" and case.execution_mode == "multiprocessing"
        )
        benchmark = getattr(benchmark_parameter_sweeps, case.benchmark_class)()

        # Use a small raster but follow ASV's normal setup/run/teardown order, including its real worker process
        parameter = 32
        try:
            benchmark.setup(parameter)
            benchmark.time_operation(parameter)
        finally:
            benchmark.teardown(parameter)


class TestVariographyWorkflows:
    """Test module for running the pair and variogram benchmark workflows."""

    def test_variogram_pairs__runs(self) -> None:
        """Checks that prepared pairs can be reduced into populated variogram bins."""

        # Prepare complete pairs across the same distance range used by the benchmark
        pairs = prepare_variogram_pairs(200)
        edges = np.geomspace(1, 1024, 9)

        # Build a variogram through the public API and check that every prepared pair reaches one bin
        result = Variogram.from_pairs(
            pairs,
            bins=edges,
            estimator=lambda differences: float(np.mean(differences**2) / 2),
        )
        assert pairs.sizes["pair"] == 200
        assert result.counts.sum() == 200
        assert np.isfinite(result.semivariance).all()

    def test_pair_pointcloud__runs(self) -> None:
        """Checks that the point benchmark fixture produces the requested number of usable pairs."""

        # Build a small irregular point cloud and draw pairs with the benchmark's fixed random seed
        points = prepare_pair_pointcloud(200)
        pairs = points.pairsample(n_pairs=20, min_distance=1, max_distance=10, random_state=42)

        # Pair values have two endpoints and a finite distance for each requested pair
        assert pairs.sizes == {"pair": 20, "endpoint": 2}
        assert np.isfinite(pairs["value"]).all()
        assert np.isfinite(pairs["distance"]).all()


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestPairRasterChunked:
    """Test module for running raster pair sampling with Dask input."""

    def test_pair_raster__dask_matches_eager(self) -> None:
        """Checks that the Dask fixture stays lazy and returns the same pairs as the eager fixture."""

        import dask.array as da

        # Prepare the same raster in memory and as Dask data without loading the Dask values
        eager = prepare_pair_raster(32, "eager")
        lazy = prepare_pair_raster(32, "dask")
        assert isinstance(lazy.data, da.Array)
        assert not lazy._obj._in_memory

        # Draw the same 20 pairs from both inputs and compare the complete public pair datasets
        options = {"n_pairs": 20, "min_distance": 1, "max_distance": 16, "random_state": 42}
        expected = eager.pairsample(**options)
        result = lazy.pairsample(**options)
        assert not result.chunks
        xr.testing.assert_equal(result, expected)


class TestGdalCommands:
    """Test module for building GDAL commands used by external benchmark comparisons."""

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command__essential_arguments(
        self, operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that every GDAL comparison names its executable, input, output and cache size."""

        # Return command names directly because this smoke test builds arguments without running GDAL
        monkeypatch.setattr("benchmarks.gdal_comparison.commands._require_command", lambda name: name)
        config = BenchmarkConfig(shape=(64, 96), chunks=(32, 32), directory=str(tmp_path))

        # Build the command with one recognizable source path for each supported input type
        comparison = build_gdal_command(
            operation,  # type: ignore[arg-type]
            config,
            raster_file="source-raster.tif",
            vector_file="source-vector.gpkg",
            point_file="source-points.gpkg",
        )
        command = comparison.command

        # These fields are enough to catch a command wired to the wrong tool, source, output or cache setting
        expected_executable = {
            "clip": "gdalwarp",
            "reproject": "gdalwarp",
            "polygonize": "gdal_polygonize.py",
            "rasterize": "gdal_rasterize",
            "grid": "gdal_grid",
        }[operation]
        expected_source = {
            "clip": "source-raster.tif",
            "reproject": "source-raster.tif",
            "polygonize": "source-raster.tif",
            "rasterize": "source-vector.gpkg",
            "grid": "source-points.gpkg",
        }[operation]
        cache_index = command.index("GDAL_CACHEMAX")
        assert command[0] == expected_executable
        assert expected_source in command
        if operation == "clip":
            assert "source-vector.gpkg" in command
        if operation in ("clip", "reproject"):
            warp_memory_index = command.index("-wm")
            assert command[warp_memory_index + 1] == str(_warp_memory_limit_mb(config))
        assert comparison.output_file in command
        assert command[cache_index + 1] == str(config.gdal_cachemax_mb)


class TestPdalCommands:
    """Test module for building PDAL pipelines used by external benchmark comparisons."""

    @pytest.mark.parametrize("operation", PDAL_COMPARISON_OPERATIONS)
    @pytest.mark.parametrize("driver", ["GPKG", "LAS", "LAZ"])
    def test_comparison_command__essential_stages(
        self,
        operation: str,
        driver: Literal["GPKG", "LAS", "LAZ"],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Checks that each PDAL comparison reads the raster and writes the expected point output."""

        # Return the command name directly because this test checks the pipeline without running PDAL
        monkeypatch.setattr("benchmarks.pdal_comparison.commands._require_command", lambda name: name)
        config = BenchmarkConfig(
            shape=(64, 96),
            chunks=(32, 32),
            subsample_size=17,
            directory=str(tmp_path),
            point_output_driver=driver,
        )

        # Build the pipeline and read the JSON passed to the PDAL command
        comparison = build_pdal_command(operation, config, raster_file="source-raster.tif")  # type: ignore[arg-type]
        pipeline = json.loads(Path(comparison.pipeline_file).read_text(encoding="utf-8"))["pipeline"]
        stage_types = [stage["type"] for stage in pipeline]

        # Conversion keeps every point, while subsampling randomizes first and keeps the requested count
        writer = "writers.ogr" if driver == "GPKG" else "writers.las"
        point_stages = ["readers.gdal"]
        if driver in ("LAS", "LAZ"):
            point_stages.append("filters.ferry")
            assert pipeline[1]["dimensions"] == "band_1=>Z"
        expected_stages = [*point_stages, writer]
        if operation == "subsample":
            expected_stages = [*point_stages, "filters.randomize", "filters.head", writer]
            assert pipeline[-3]["seed"] == 42
            assert pipeline[-2]["count"] == config.subsample_size
        assert stage_types == expected_stages

        # Both paths use the configured cache, source raster and point output
        assert comparison.command == ["pdal", "pipeline", comparison.pipeline_file]
        assert pipeline[0]["filename"] == "source-raster.tif"
        assert pipeline[0]["gdalopts"] == [f"GDAL_CACHEMAX={config.gdal_cachemax_mb}"]
        assert pipeline[-1]["filename"] == comparison.output_file
        assert Path(comparison.output_file).suffix == f".{driver.lower()}"
        if driver == "GPKG":
            assert pipeline[-1]["ogrdriver"] == "GPKG"
        else:
            assert pipeline[-1]["compression"] == (driver == "LAZ")
            assert pipeline[-1]["extra_dims"] == "all"
