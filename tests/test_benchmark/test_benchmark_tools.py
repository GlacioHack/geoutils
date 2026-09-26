"""Minimal tests for benchmark workflows, reports and external CLI commands."""

from __future__ import annotations

import hashlib
import json
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import rasterio as rio

from benchmarks.asv_suite import benchmarks as generated_benchmarks
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
    render_documentation_snapshot,
)
from benchmarks.comparisons.flox import (
    compute_geoutils_grouped_stats,
    prepare_grouped_inputs,
)
from benchmarks.comparisons.flox import (
    grouped_stats as flox_grouped_stats,
)
from benchmarks.comparisons.gdal import (
    COMPARISON_OPERATIONS,
    _warp_memory_limit_mb,
    build_gdal_command,
)
from benchmarks.comparisons.pdal import (
    PDAL_COMPARISON_OPERATIONS,
    PdalComparisonOperation,
    build_pdal_command,
)
from benchmarks.workflows.config import RuntimeConfig
from benchmarks.workflows.core import Case
from benchmarks.workflows.operations import (
    BENCHMARK_BY_CLASS,
    BENCHMARK_BY_ID,
    BENCHMARK_CASE_BY_CLASS,
    BENCHMARK_CASES,
    BENCHMARKS,
    COMPARISONS,
    OPERATION_BY_NAME,
    OPERATION_MODULES,
    collect_operation_modules,
    discover_operation_modules,
    format_api_label,
)
from benchmarks.workflows.runner import BenchmarkRunner


class TestComparisonReport:
    """
    Test module for benchmark registration and report generation.

    Those are minimal tests to ensure changes to ``benchmarks/`` don't break the routines.
    Even if ASV runs quick checks of the benchmarking setup through CI, it's easier to have a detailed traceback here
    through Pytest for some aspects.
    We especially tests our custom routines/rendering for the benchmark webpage, comparisons to external refs (e.g.
    GDAL CLI), and across variables (raster/point size, chunk size, etc) and categories (e.g. in-memory/Dask/MP, method,
    etc) of interest.
    """

    def test_benchmark_registry__generated_classes_exist(self) -> None:
        """Checks that every registered benchmark and plotted series has a generated ASV class."""

        # Gather the GeoUtils/external classes registered for ASV and the classes referenced by report plots
        registered = set(BENCHMARK_CASE_BY_CLASS)
        plotted = {class_name for comparison in COMPARISONS for _, class_name in comparison.series}

        # Importing benchmarks.py should create every class needed by ASV and the report
        assert BENCHMARK_CASES and COMPARISONS
        assert plotted <= registered
        assert all(hasattr(generated_benchmarks, class_name) for class_name in registered)

    def test_benchmark_registry__generated_class_names_stable(self) -> None:
        """Checks that generated ASV identifiers keep function fields separate from their varying input."""

        # Hash the sorted names to keep the exact 118-class identifier check compact and order independent
        class_names = sorted(
            benchmark.benchmark_class(case)
            for benchmark in BENCHMARKS
            if benchmark.parameter_name is not None
            for case in benchmark.cases
        )
        digest = hashlib.sha256("\n".join(class_names).encode()).hexdigest()

        # A changed name would split ASV history even when the underlying operation remained the same
        assert digest == "608dbbf7be17dbd67d5b42e9953d1568e12281d06bd9bd3cf64959ba7517ba38"

    def test_operation_discovery__deterministic_modules(self) -> None:
        """Checks that operation discovery returns the same modules in their stable report order."""

        # Repeat package discovery rather than reusing the modules collected during the first import
        discovered = discover_operation_modules()

        # Every run should return the same local modules in the order used for sweeps and comparison plots
        assert tuple(module.__name__ for module in discovered) == tuple(module.__name__ for module in OPERATION_MODULES)
        assert [getattr(module, "ORDER", 100) for module in discovered] == sorted(
            getattr(module, "ORDER", 100) for module in discovered
        )

    def test_operation_discovery__error_duplicate_benchmark_id(self) -> None:
        """Checks that two operation modules cannot register the same benchmark identifier."""

        # Present one real sweep through two small module-like objects to isolate duplicate validation
        duplicate_modules = (SimpleNamespace(BENCHMARKS=(BENCHMARKS[0],)),) * 2

        # Reject the collision before generated classes or report mappings can silently replace each other
        with pytest.raises(ValueError, match=f"Duplicate benchmark ID: {BENCHMARKS[0].id}"):
            collect_operation_modules(duplicate_modules)

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
    """Test module for scheduled inputs and benchmark configurations."""

    def test_sweeps__representative_configs(self) -> None:
        """Checks representative updates for strategy, grouped and point sweeps."""

        # Select cases whose configuration depends on their method, layout or point count
        gridding = BENCHMARK_BY_ID["grid-raster-size"]
        idw_case = next(case for case in gridding.cases if case.method == "idw" and case.execution == "inmem")
        interleaved = BENCHMARK_BY_ID["grouped-stats-interleaved-chunk-size"]
        grouped_case = interleaved.cases[0]
        pointcloud = BENCHMARK_BY_ID["to-pointcloud-raster-size"]
        point_case = pointcloud.cases[0]

        # These values cover method-specific support, uneven chunks, repeated groups and point inputs
        grid_config = gridding.make_config(8_000, idw_case)
        grouped_config = interleaved.make_config(500, grouped_case)
        point_config = pointcloud.make_config(1_000, point_case)
        assert (grid_config.shape, grid_config.chunks, grid_config.value("grid_dist_nodata_pixel")) == (
            (8_000, 8_000),
            (1_000, 1_000),
            16,
        )
        assert (grouped_config.shape, grouped_config.chunks, grouped_config.value("grouped_layout")) == (
            (2_000, 2_000),
            (500, 500),
            "interleaved",
        )
        assert (point_config.shape, point_config.chunks) == ((1_000, 1_000), (1_000, 1_000))

    def test_comparison__derived_identity_and_workload(self) -> None:
        """Checks that sweep cases provide report metadata and workload sizes without duplicate text."""

        # Select a comparison with raster, chunk and vector fixture dimensions
        comparison = next(item for item in COMPARISONS if item.slug == "clip-raster-size")

        # The operation and parameter axis name the sweep, while its config supplies every displayed workload value
        assert comparison.benchmark.id == "clip-raster-size"
        assert comparison.operation == "clip"
        assert comparison.by == "execution"
        assert comparison.workload(8_000) == ("8,000 × 8,000 raster; 1,000 × 1,000 chunks; 51 × 51 vector features")

    def test_comparison__output_formats_group_matching_implementations(self) -> None:
        """Checks that each GeoUtils point format stays beside the matching PDAL format in report series."""

        # Select the output-format comparison whose benchmark stores GeoUtils cases before implementation cases
        comparison = next(item for item in COMPARISONS if item.slug == "subsample-las-laz-size")

        # Keep the established report order while treating every reference implementation as a normal case
        assert tuple(label for label, _ in comparison.series) == (
            "GeoUtils LAS",
            "PDAL LAS",
            "GeoUtils LAZ",
            "PDAL LAZ",
        )

    @pytest.mark.parametrize(
        ("class_name", "expected_label"),
        (
            ("reproject_nearest_rasterio_inmem__rastersize", ".reproject(resampling='nearest')"),
            ("polygonize_rasterio_labelstitch_dask__rastersize", ".polygonize(strategy='label_stitch')"),
            (
                "stats_moments_numpy_dense_dask__rastersize",
                "stats(statistics=['mean', 'std', 'min', 'max'], strategy='dense')",
            ),
        ),
    )
    def test_api_label__matches_execution_options(self, class_name: str, expected_label: str) -> None:
        """Checks that generated labels use the public argument names passed by operation handlers."""

        # Resolve the structured case instead of deriving options from its generated class name
        case = BENCHMARK_CASE_BY_CLASS[class_name]
        benchmark = BENCHMARK_BY_CLASS[class_name]

        # Execution mode stays separate from the public call while method, engine and strategy remain visible
        assert format_api_label(benchmark.operation, case) == expected_label
        assert getattr(generated_benchmarks, class_name).pretty_name == expected_label

    def test_comparison_references__use_normal_cases(self) -> None:
        """Checks that Flox, GDAL and PDAL references are normal benchmark cases."""

        # Flox measures prepared arrays directly; GDAL and PDAL execute through their command runners
        flox_benchmark = BENCHMARK_BY_ID["grouped-flox-raster-size"]
        gdal_benchmark = BENCHMARK_BY_ID["rasterize-raster-size"]
        pdal_benchmark = BENCHMARK_BY_ID["to-pointcloud-raster-size"]

        # Flox is a public stats() backend, while GDAL and PDAL remain external command references
        assert {case.implementation for case in flox_benchmark.cases} == {"geoutils", "flox"}
        assert {case.implementation for case in gdal_benchmark.cases} == {"geoutils", "gdal"}
        assert {case.implementation for case in pdal_benchmark.cases} == {"geoutils", "pdal"}


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestBenchmarkRunner:
    """Test module for bounded operation outputs produced by BenchmarkRunner."""

    @pytest.mark.parametrize("execution_mode", ["inmem", "dask", "multiprocessing"])
    def test_clip__masks_outside_fixture_polygons(
        self,
        execution_mode: Literal["inmem", "dask", "multiprocessing"],
        tmp_path: Path,
    ) -> None:
        """Checks that the clipping benchmark keeps polygon interiors and masks the surrounding raster."""

        if execution_mode == "dask":
            pytest.importorskip("distributed")

        # Use several polygons across multiple chunks so worker modes process inside and outside cells
        config = RuntimeConfig(
            shape=(64, 64),
            chunks=(32, 32),
            directory=str(tmp_path / execution_mode),
            workload={"vector_features_per_axis": 3, "operation": "clip"},
        )

        # Run the complete benchmark workflow and inspect one kept center cell and one clipped corner
        operation = OPERATION_BY_NAME["clip"]
        case = Case(execution=execution_mode, options={"operation": "clip"})
        with BenchmarkRunner(operation, case, config) as runner:
            result = runner.run(profile=False)
        assert result.output_file is not None
        with rio.open(result.output_file) as dataset:
            values = dataset.read(1)
            nodata = dataset.nodata

        # The odd polygon grid covers the raster center, while every fixture polygon stays away from the corners
        assert result.value == config.raster_value
        assert values[values.shape[0] // 2, values.shape[1] // 2] == config.raster_value
        assert not np.isfinite(values[0, 0]) or values[0, 0] == nodata


def test_grouped_reference__matches_geoutils() -> None:
    """Checks that direct Flox and public GeoUtils grouped statistics return the same in-memory table."""

    pytest.importorskip("flox")

    # Prepare one small shared input and run each implementation directly
    inputs = prepare_grouped_inputs(16, 2, "interleaved", "inmem")
    expected = compute_geoutils_grouped_stats(*inputs)
    result = flox_grouped_stats(*inputs, use_dask=False)

    # Matching labelled tables establish that the external reference measures the same calculation
    assert result.equals(expected)


class TestGdalCommands:
    """Test module for building GDAL commands used by external benchmark comparisons."""

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command__essential_arguments(
        self, operation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that every GDAL comparison names its executable, input, output and cache size."""

        # Return command names directly because this smoke test builds arguments without running GDAL
        monkeypatch.setattr("benchmarks.comparisons.gdal._require_command", lambda name: name)
        config = RuntimeConfig(shape=(64, 96), chunks=(32, 32), directory=str(tmp_path))

        # Build the command with one recognizable source path for each supported input type
        comparison = build_gdal_command(
            operation,  # type: ignore[arg-type]
            Case(method="nearest" if operation in ("reproject", "grid") else None),
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
        operation: PdalComparisonOperation,
        driver: Literal["GPKG", "LAS", "LAZ"],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Checks that each PDAL comparison reads the raster and writes the expected point output."""

        # Return the command name directly because this test checks the pipeline without running PDAL
        monkeypatch.setattr("benchmarks.comparisons.pdal._require_command", lambda name: name)
        config = RuntimeConfig(
            shape=(64, 96),
            chunks=(32, 32),
            directory=str(tmp_path),
            workload={"subsample_size": 17},
        )

        # Build the pipeline and read the JSON passed to the PDAL command
        comparison = build_pdal_command(
            operation,
            Case(output_driver=driver),
            config,
            raster_file="source-raster.tif",
        )
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
            assert pipeline[-2]["count"] == config.value("subsample_size")
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
