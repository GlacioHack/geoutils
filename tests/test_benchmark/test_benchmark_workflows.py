"""Test the shared benchmark workflows, reports and GDAL command adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio

from benchmarks.asv_suite.comparisons import COMPARISONS
from benchmarks.asv_suite.render_results import (
    COMPARISON_REPORT_DIRECTORY,
    DOCUMENTATION_DATA,
    DOCUMENTATION_MEMORY_PLOT,
    DOCUMENTATION_TIME_PLOT,
    collect_implementation_measurements,
    render_comparisons,
    render_documentation_snapshot,
    select_asv_result,
)
from benchmarks.gdal_comparison.commands import (
    COMPARISON_OPERATIONS,
    build_gdal_command,
    build_gdal_grid_command,
)
from benchmarks.gdal_comparison.runner import GdalRunner
from benchmarks.workflows.registry import (
    OPERATION_BENCHMARK_CASES,
    OPERATION_BY_NAME,
    OPERATION_CASES,
    split_operation_case,
)
from benchmarks.workflows.runner import BenchmarkConfig, BenchmarkRunner
from geoutils.interface.gridding import GriddingMethod, _grid_pointcloud


class TestOperationRegistry:
    """Check that benchmark identifiers retain complete and unambiguous coverage."""

    def test_registered_backend_cases_are_unique(self) -> None:
        """Represent every declared operation-backend pair exactly once."""

        # Reconstruct expected cases independently from the exported flat list
        expected = {(backend, case.operation) for case in OPERATION_CASES for backend in case.backends}
        parsed = {split_operation_case(case_name) for case_name in OPERATION_BENCHMARK_CASES}

        assert len(OPERATION_BENCHMARK_CASES) == len(set(OPERATION_BENCHMARK_CASES))
        assert parsed == expected


class TestComparisonReport:
    """Check that saved ASV comparisons become reusable numeric reports and plots."""

    class FakeResult:
        """Provide the small part of the ASV Results interface used by the renderer."""

        commit_hash = "0123456789abcdef"
        date = 1_700_000_000_000
        params = {"machine": "test-machine", "cpu": "test-cpu"}
        started_at = {"benchmark": 1_700_000_001_000}
        _env_name = "existing-py3.12"

        def __init__(self) -> None:
            """Build matching time and memory values for every comparison class."""

            self.values = {}
            self.parameters = {}
            for comparison in COMPARISONS:
                for implementation_index, (_, class_name) in enumerate(comparison.series, start=1):
                    # Test parameters are stored as repr strings like real ASV results
                    parameters = [["256", "512", "1024"]]
                    prefix = f"asv_suite.comparisons.{class_name}"
                    self.parameters[f"{prefix}.time_operation"] = parameters
                    self.parameters[f"{prefix}.track_end_to_end_time_s"] = parameters
                    self.parameters[f"{prefix}.track_peak_process_tree_rss_mb"] = parameters
                    self.values[f"{prefix}.time_operation"] = [
                        implementation_index * value / 1000 for value in (1, 2, 4)
                    ]
                    self.values[f"{prefix}.track_end_to_end_time_s"] = [
                        implementation_index * value / 100 for value in (1, 2, 4)
                    ]
                    self.values[f"{prefix}.track_peak_process_tree_rss_mb"] = [
                        implementation_index * value for value in (100, 110, 120)
                    ]

        def get_all_result_keys(self) -> Any:
            """Return available benchmark keys like ASV Results."""

            return self.values.keys()

        def get_result_params(self, key: str) -> list[list[str]]:
            """Return the one stored parameter dimension."""

            return self.parameters[key]

        def get_result_value(self, key: str, params: list[list[str]]) -> list[float]:
            """Return measurements already aligned to the requested parameters."""

            assert params == self.parameters[key]
            return self.values[key]

    def test_collect_implementation_measurements(self) -> None:
        """Pair both time boundaries and peak RAM for every implementation."""

        # Every declared line and parameter must remain an independent numeric record
        records = collect_implementation_measurements(self.FakeResult())
        expected_records = sum(len(comparison.series) * 3 for comparison in COMPARISONS)
        assert len(records) == expected_records
        assert {record.comparison for record in records} == {comparison.slug for comparison in COMPARISONS}
        assert {record.implementation for record in records} == {
            "Eager",
            "Dask",
            "Multiprocessing",
            "Eager (Numba)",
            "SciPy",
            "Numba",
            "GDAL",
        }
        assert all(isinstance(record.parameter, int) for record in records)

    def test_select_asv_result__skip_incomplete(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Keep the latest complete result when a newer ASV run was interrupted."""

        complete = self.FakeResult()
        complete.started_at = {"benchmark": 1_700_000_001_000}
        incomplete = self.FakeResult()
        incomplete.started_at = {"benchmark": 1_700_000_002_000}

        # Removing one series reproduces an interrupted comparison run
        incomplete.values.pop(next(iter(incomplete.values)))
        monkeypatch.setattr("asv.results.iter_results", lambda _: iter((complete, incomplete)))

        assert select_asv_result(tmp_path) is complete

    def test_render_comparisons(self, tmp_path: Path) -> None:
        """Write one site index, numeric files and one SVG per comparison."""

        pytest.importorskip("matplotlib")

        # The root remains the single entry point for local artifacts and GitHub Pages
        render_comparisons(self.FakeResult(), tmp_path)
        assert (tmp_path / "index.html").is_file()
        assert "comparisons/index.html" in (tmp_path / "index.html").read_text(encoding="utf-8")
        report_directory = tmp_path / COMPARISON_REPORT_DIRECTORY
        assert (report_directory / "index.html").is_file()
        assert (report_directory / "comparisons.json").is_file()
        assert (report_directory / "comparisons.csv").is_file()
        assert {path.name for path in report_directory.glob("*.svg")} == {
            f"{comparison.slug}.svg" for comparison in COMPARISONS
        }

    def test_render_documentation_snapshot(self, tmp_path: Path) -> None:
        """Write two concise graphics with their exact measurements and provenance."""

        pytest.importorskip("matplotlib")

        # Documentation files derive from the same records as the detailed ASV website
        render_documentation_snapshot(self.FakeResult(), tmp_path)
        assert (tmp_path / DOCUMENTATION_TIME_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_MEMORY_PLOT).is_file()
        assert (tmp_path / DOCUMENTATION_DATA).is_file()

        snapshot = json.loads((tmp_path / DOCUMENTATION_DATA).read_text(encoding="utf-8"))
        assert snapshot["metadata"]["commit"] == "0123456789abcdef"
        assert snapshot["metadata"]["machine"]["machine"] == "test-machine"


class TestGdalComparison:
    """Run every GDAL adapter on compact deterministic source files."""

    @pytest.mark.parametrize("operation", COMPARISON_OPERATIONS)
    def test_comparison_command_output(self, operation: str, tmp_path: Path) -> None:
        """Create an equivalent GDAL output with the expected small fingerprint."""

        # Prepare the same file inputs used by canonical comparison jobs
        config = BenchmarkConfig(
            shape=(64, 64),
            chunks=(32, 32),
            polygon_regions_per_axis=3,
            vector_features_per_axis=3,
            point_features_per_axis=3,
            directory=str(tmp_path),
        )
        source_runner = BenchmarkRunner("dask", config).prepare_sources()
        comparison = build_gdal_command(
            operation,  # type: ignore[arg-type]
            config,
            raster_file=(source_runner.polygon_raster_file if operation == "polygonize" else source_runner.raster_file),
            vector_file=source_runner.vector_file,
            point_file=source_runner.point_file,
        )

        # The runner executes without a shell and validates the complete output
        runner = GdalRunner(operation, config, source_runner)  # type: ignore[arg-type]
        result = runner.run()
        expected = config.polygon_regions_per_axis**2 if operation == "polygonize" else 1
        assert comparison.command == runner.comparison.command
        if operation == "reproject":
            # The GDAL baseline must use the exact transform applied by every GeoUtils backend
            assert comparison.command[comparison.command.index("-et") + 1] == "0"
            assert "XSCALE=1" in comparison.command
            assert "YSCALE=1" in comparison.command
        assert np.isclose(result.value, expected)
        assert Path(result.output_file).is_file()
        assert result.peak_process_tree_rss_mb > 0

    @pytest.mark.parametrize(("geoutils_method", "gdal_algorithm"), [("nearest", "nearest"), ("linear", "linear")])
    def test_interpolation_gridding_matches_gdal(
        self, geoutils_method: GriddingMethod, gdal_algorithm: str, tmp_path: Path
    ) -> None:
        """Match complete GDAL nearest and linear outputs for an irregular point cloud."""

        # Unequal positions and values expose axis rescaling and nearest-neighbor differences
        points = gpd.GeoDataFrame(
            {"z": [2.0, 8.0, 4.0, 10.0, 6.0]},
            geometry=gpd.points_from_xy(x=[0.0, 37.0, 4.0, 40.0, 17.0], y=[0.0, 0.2, 3.6, 4.0, 2.3]),
            crs=32631,
        )
        point_file = tmp_path / "interpolation-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        grid_coords = (np.arange(0, 50, 10, dtype=float), np.arange(5, dtype=float))

        # Infinite nearest support and linear interpolation without extrapolation match GDAL radius zero
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=geoutils_method,
            dist_nodata_pixel=float("inf"),
            engine="scipy",
        )
        output_file = tmp_path / f"gdal-{gdal_algorithm}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm=gdal_algorithm,  # type: ignore[arg-type]
            bounds=(-5.0, -0.5, 45.0, 4.5),
            shape=(5, 5),
            radius=(0.0, 0.0),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    @pytest.mark.parametrize(
        ("geoutils_method", "gdal_algorithm"),
        [
            ("idw", "invdist"),
            ("mean", "average"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("range", "range"),
            ("count", "count"),
            ("average_distance", "average_distance"),
            ("average_distance_pts", "average_distance_pts"),
        ],
    )
    def test_circular_gridding_matches_gdal(
        self, geoutils_method: GriddingMethod, gdal_algorithm: str, tmp_path: Path
    ) -> None:
        """Match GDAL values and nodata cells for every shared circular method."""

        # Two unequal values exercise all statistics while an invalid value checks GDAL's omission rule
        points = gpd.GeoDataFrame(
            {"z": [2.0, np.nan, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / "circular-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        x_coords = np.arange(5, dtype=float)
        y_coords = np.array([0.0])

        # GeoUtils expresses the support ellipse in output pixels
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(x_coords, y_coords),
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling=geoutils_method,
            dist_nodata_pixel=1.1,
        )

        # GDAL receives the equivalent support in coordinate units and the same cell centers
        output_file = tmp_path / f"gdal-{gdal_algorithm}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm=gdal_algorithm,  # type: ignore[arg-type]
            bounds=(-0.5, -0.5, 4.5, 0.5),
            shape=(1, 5),
            radius=(1.1, 1.1),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    def test_idw_exact_points_match_gdal_minimum_point_behavior(self, tmp_path: Path) -> None:
        """Give exact IDW points precedence over minimum neighbor counts like GDAL."""

        # Exact edge cells have one neighbor while the central cell has two
        points = gpd.GeoDataFrame(
            {"z": [2.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0], y=[0.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / "idw-points.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(np.arange(3, dtype=float), np.array([0.0])),
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling="idw",
            dist_nodata_pixel=1.1,
            min_points=2,
        )

        output_file = tmp_path / "gdal-idw-min-points.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm="invdist",
            bounds=(-0.5, -0.5, 2.5, 0.5),
            shape=(1, 3),
            radius=(1.1, 1.1),
            min_points=2,
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)

    @pytest.mark.parametrize("engine", ["scipy", "numba"])
    def test_idw_anisotropic_grid_matches_gdal(self, engine: str, tmp_path: Path) -> None:
        """Weight IDW values by coordinate distance on an anisotropic output grid like GDAL."""

        if engine == "numba":
            pytest.importorskip("numba")

        # One X pixel is ten times larger than one Y pixel so scaled and coordinate distances differ
        points = gpd.GeoDataFrame(
            {"z": [0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 10.0], y=[1.0, 0.0]),
            crs=32631,
        )
        point_file = tmp_path / f"anisotropic-idw-{engine}.gpkg"
        points.to_file(point_file, layer="source-points", driver="GPKG")
        expected, _ = _grid_pointcloud(
            points,
            grid_coords=(np.arange(0, 30, 10, dtype=float), np.arange(3, dtype=float)),
            data_column_name="z",
            resampling="idw",
            dist_nodata_pixel=1.1,
            engine=engine,  # type: ignore[arg-type]
        )

        # GDAL receives the same support radius converted from pixels to coordinate units
        output_file = tmp_path / f"gdal-anisotropic-idw-{engine}.tif"
        command = build_gdal_grid_command(
            str(point_file),
            str(output_file),
            algorithm="invdist",
            bounds=(-5.0, -0.5, 25.0, 2.5),
            shape=(3, 3),
            radius=(11.0, 1.1),
        )
        subprocess.run(command.command, check=True, capture_output=True, text=True)
        with rio.open(output_file) as dataset:
            actual = dataset.read(1, masked=True).filled(np.nan)

        assert np.allclose(actual, expected, equal_nan=True)


@pytest.mark.allow_logging_warnings
class TestBenchmarkRunner:
    """Exercise every shared operation on compact data before expensive jobs run."""

    def test_multiprocessing_worker_operations(self) -> None:
        """Run every multiprocessing operation through a real worker process."""

        # One compact fixture catches callables that cannot cross process boundaries
        config = BenchmarkConfig(
            shape=(256, 256),
            chunks=(128, 128),
            ninterp=32,
            subsample_size=32,
        )
        operations = [case.operation for case in OPERATION_CASES if "multiprocessing" in case.backends]
        with BenchmarkRunner("multiprocessing", config) as runner:
            for operation in operations:
                value = runner._execute(operation)
                assert np.isclose(value, OPERATION_BY_NAME[operation].expected_value), operation

    def test_eager_comparison_operations(self) -> None:
        """Run every eager comparison on compact in-memory data."""

        # Eager coverage checks the common file-to-file computation paths without workers
        config = BenchmarkConfig(
            shape=(64, 64),
            chunks=(32, 32),
            ninterp=32,
            polygon_regions_per_axis=3,
            vector_features_per_axis=3,
            point_features_per_axis=3,
        )
        with BenchmarkRunner("eager", config) as runner:
            for operation in COMPARISON_OPERATIONS:
                value = runner._execute(operation)
                expected = config.polygon_regions_per_axis**2 if operation == "polygonize" else 1
                assert np.isclose(value, expected), operation

            # The large data suite selects the same runner path for both circular gridding methods
            for resampling in ("idw", "mean"):
                runner.config.grid_resampling = resampling
                runner.config.grid_dist_nodata_pixel = 2
                assert np.isclose(runner._execute("grid"), 1), resampling

    def test_dask_worker_operations(self) -> None:
        """Run every Dask operation through one local distributed worker."""

        pytest.importorskip("dask")
        pytest.importorskip("distributed")
        pytest.importorskip("dask_geopandas")

        # Reuse one worker so compact regression coverage remains inexpensive
        config = BenchmarkConfig(
            shape=(256, 256),
            chunks=(128, 128),
            memory_limit="512MB",
            ninterp=32,
            subsample_size=32,
        )
        operations = [case.operation for case in OPERATION_CASES if "dask" in case.backends]
        with BenchmarkRunner("dask", config) as runner:
            for operation in operations:
                value = runner._execute(operation)
                assert np.isclose(value, OPERATION_BY_NAME[operation].expected_value), operation
