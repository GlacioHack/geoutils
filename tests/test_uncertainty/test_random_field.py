"""Tests for random error fields over rasters, point clouds, and explicit coordinates."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.transform import from_origin

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.stats.variography import VariogramModel


class TestRandomField:
    """
    Test module for random fields.

    Tests for chunked backends (Dask/MP) are further below in TestRandomFieldChunked,
    while errors/warnings raised are in TestRandomFieldErrors.
    """

    def test_generate_random_field__match_raster(self) -> None:
        """Checks that random fields have the same shape, coordinates, and nodata as the input raster."""

        # We create a synthetic raster and error structure
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True
        values = np.ma.masked_array(np.ones((2, 3)), mask=mask)
        raster = gu.Raster.from_array(values, transform=from_origin(10, 20, 2, 2), crs=32606)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check equal shapes/coords/nodata through all three APIs
        first = gu.uncertainty.random_field(structure, like=raster, random_state=5)
        second = structure.generate_random_field(raster, random_state=5)
        method_result = raster.random_field(structure, random_state=5)
        assert isinstance(first, gu.Raster) and isinstance(second, gu.Raster) and isinstance(method_result, gu.Raster)
        assert first.shape == raster.shape and first.transform == raster.transform and first.crs == raster.crs
        np.testing.assert_array_equal(first.get_mask(), mask)
        np.testing.assert_array_equal(first.data, second.data)
        np.testing.assert_array_equal(first.data, method_result.data)

    def test_random_field__match_point(self) -> None:
        """Checks that random_field has the same corrds as input point cloud."""

        # A synthetic point cloud and error structure
        points = gu.PointCloud.from_xyz([0, 2, 1, 4], [1, 0, 3, 2], [10, 11, 12, 13], crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check they match
        result = points.random_field(structure, random_state=8)
        expected = gu.uncertainty.random_field(structure, like=points, random_state=8)
        np.testing.assert_array_equal(result.geometry.x, points.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, points.geometry.y)
        np.testing.assert_array_equal(result.data, expected.data)

    def test_random_field__duplicate_point_labels_use_row_identity(self) -> None:
        """Checks that duplicate point labels still draw independent errors for each observation."""

        # Duplicate table labels cannot identify the two source errors
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        points.ds.index = pd.Index(["same", "same"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Row numbers distinguish the independent error of each point
        result = points.random_field(structure, random_state=4)
        coordinates = np.column_stack((points.geometry.x.to_numpy(), points.geometry.y.to_numpy()))
        expected = structure.generate_random_field(coordinates=coordinates, random_state=4)
        np.testing.assert_array_equal(result.data, expected)
        assert result.data[0] != result.data[1]

    def test_random_field__raster_predictors_scale_error_by_location(self) -> None:
        """Checks that a raster predictor gives each pixel its own interpolated error magnitude."""

        # Slopes 0, 0.5, and 1 interpolate magnitudes 1, 1.5, and 2
        transform = from_origin(0, 2, 1, 1)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=transform, crs=32606)
        slope = gu.Raster.from_array(np.array([[0.0, 0.5], [1.0, 0.5]]), transform=transform, crs=32606)
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        variable = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        unit = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Same seed gives the same standard-normal draw before spatial magnitudes are applied
        result = raster.random_field(variable, predictors={"slope": slope}, random_state=5)
        reference = raster.random_field(unit, random_state=5)
        expected = reference.to_nanarray() * np.array([[1.0, 1.5], [2.0, 1.5]])
        np.testing.assert_allclose(result.to_nanarray(), expected)

    def test_random_field__scalar_predictor_applies_everywhere(self) -> None:
        """Checks that one predictor value sets the same error magnitude at every raster pixel."""

        # Slope 0.5 lies halfway between tabulated magnitudes 1 and 2
        raster = gu.Raster.from_array(np.ones((2, 3)), transform=from_origin(0, 2, 1, 1), crs=32606)
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        variable = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        unit = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Same seeded normal field scaled by the interpolated magnitude 1.5
        result = raster.random_field(variable, predictors={"slope": 0.5}, random_state=5)
        reference = raster.random_field(unit, random_state=5)
        np.testing.assert_allclose(result.to_nanarray(), 1.5 * reference.to_nanarray())

    def test_random_field__point_predictor_column_scales_each_error(self) -> None:
        """Checks that a named point cloud column supplies each observation's error magnitude."""

        # Two points with slope values at the tabulated magnitudes 1 and 2
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        points.ds["slope"] = [0.0, 1.0]
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        variable = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        unit = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Compare seeded errors before and after their point-specific scaling
        result = points.random_field(variable, predictors={"slope": "slope"}, random_state=5)
        reference = points.random_field(unit, random_state=5)
        np.testing.assert_allclose(result.data, reference.data * [1, 2])

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__single_coordinate_axis(self) -> None:
        """Checks that a model using only X produces matching errors along each grid column."""

        # Two rows share each X coordinate but have different Y coordinates
        raster = gu.Raster.from_array(np.ones((2, 3)), transform=from_origin(0, 2, 1, 1), crs=32606)
        correlation = VariogramModel("gaussian", effective_range=3, partial_sill=1, active_dims=(0,))
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # A one-dimensional field must repeat across rows at the same X positions
        result = raster.random_field(structure, random_state=7)
        np.testing.assert_array_equal(result.to_nanarray()[0], result.to_nanarray()[1])

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__correlated_support_is_reproducible(self) -> None:
        """Checks that a correlated component draws one reproducible field across its complete support."""

        # Exponential correlation across source coordinates
        coordinates = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
        correlation = VariogramModel("exponential", effective_range=4, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])

        # Check field size and reproducibility with same seed
        first = gu.uncertainty.random_field(structure, coordinates=coordinates, random_state=3)
        second = gu.uncertainty.random_field(structure, coordinates=coordinates, random_state=3)
        assert first.shape == (8,)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.skipif(find_spec("gpytorch") is None or find_spec("torch") is None, reason="Requires GPyTorch")
    def test_random_field__gpytorch_uses_irregular_point_support(self) -> None:
        """Checks that the GPyTorch backend returns reproducible correlated values on irregular points."""

        # Irregular points with exponential model supported by GPyTorch
        coordinates = np.array([[0.0, 0.0], [0.5, 1.5], [2.0, 0.25], [3.0, 2.0]])
        correlation = VariogramModel("exponential", effective_range=3, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1.5, correlation)])

        # Check finite field and reproducibility with same seed
        first = gu.uncertainty.random_field(
            structure,
            coordinates=coordinates,
            random_state=9,
            backend="gpytorch",
        )
        second = gu.uncertainty.random_field(
            structure,
            coordinates=coordinates,
            random_state=9,
            backend="gpytorch",
        )
        assert first.shape == (4,)
        assert np.all(np.isfinite(first))
        np.testing.assert_array_equal(first, second)
        assert np.std(first) > 0

    @pytest.mark.skipif(find_spec("gpytorch") is None or find_spec("torch") is None, reason="Requires GPyTorch")
    def test_random_field__gpytorch_keeps_regular_raster_support(self) -> None:
        """Checks that GPyTorch fields have the same shape, coordinates, and missing cells as the raster."""

        # Small raster (GPyTorch builds full covariance matrix)
        values = np.ma.masked_array(np.ones((3, 4)), mask=np.eye(3, 4, dtype=bool))
        raster = gu.Raster.from_array(values, transform=from_origin(10, 20, 2, 2), crs=32606)
        correlation = VariogramModel("exponential", effective_range=4, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # Check reproducible values, original grid and mask
        first = raster.random_field(structure, random_state=5, backend="gpytorch")
        second = raster.random_field(structure, random_state=5, backend="gpytorch")
        assert first.shape == raster.shape and first.transform == raster.transform and first.crs == raster.crs
        np.testing.assert_array_equal(first.get_mask(), raster.get_mask())
        np.testing.assert_array_equal(first.data, second.data)


class TestRandomFieldChunked:
    """Test module for matching eager, Dask and multiprocessing fields across raster tiles and point rows."""

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__raster_chunk_invariance(self) -> None:
        """Checks that changing raster chunks does not change independent or correlated field realizations."""

        # Independent + correlated components on chunked grid
        raster = gu.Raster.from_array(
            np.ones((6, 7)),
            transform=from_origin(0, 6, 2, 2),
            crs=32606,
        )
        correlation = VariogramModel("gaussian", effective_range=6, partial_sill=1)
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("measurement", 0.5), gu.ErrorComponent("spatial", 2, correlation)]
        )

        # Check lazy chunks with shorter edges (same seeds across all tiles)
        expected = raster.random_field(structure, random_state=9)
        lazy = raster.random_field(structure, random_state=9, chunksizes=(4, 3))
        assert hasattr(lazy.data, "compute")
        assert lazy.data.chunks == ((4, 2), (3, 3, 1))

        # Compare exact equality of Dask field with eager result across chunk boundaries
        np.testing.assert_array_equal(np.asarray(lazy.compute()), expected.to_nanarray())

    @pytest.mark.parametrize("backend", ["gstools", "gpytorch"])
    @pytest.mark.parametrize("active_dims", [None, (0,)])
    def test_random_field__rotated_raster_chunks(
        self, active_dims: tuple[int, ...] | None, backend: Literal["gstools", "gpytorch"]
    ) -> None:
        """Checks that rotated raster chunks use matching full or selected map coordinates."""

        pytest.importorskip(backend)

        # Rotation makes X and Y depend on both row and column; final chunks are shorter
        transform = Affine(2, 0.5, 10, 0.25, -2, 20)
        raster = gu.Raster.from_array(np.ones((3, 4)), transform=transform, crs=32606)
        correlation = VariogramModel("gaussian", effective_range=5, partial_sill=1, active_dims=active_dims)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # Keep the eager input loaded while constructing a lazy field from 2 x 3 tiles
        assert not hasattr(raster.data, "compute")
        inducing_points = 64 if backend == "gpytorch" else None
        expected = raster.random_field(
            structure, random_state=7, backend=backend, gpytorch_inducing_points=inducing_points
        )
        lazy = raster.random_field(
            structure,
            random_state=7,
            chunksizes=(2, 3),
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert hasattr(lazy.data, "compute")
        assert lazy.data.chunks == ((2, 1), (3, 1))

        # Compare values at every rotated center, including the shorter edge tiles
        np.testing.assert_allclose(np.asarray(lazy.compute()), expected.to_nanarray(), rtol=0, atol=1e-12)

    @pytest.mark.skipif(find_spec("gpytorch") is None, reason="Requires GPyTorch")
    def test_random_field__gpytorch_default_inducing_grid(self) -> None:
        """Checks that a chunked GP field uses the documented default 256-point grid."""

        # A small raster has edge tiles and a correlated component
        raster = gu.Raster.from_array(np.ones((3, 4)), transform=from_origin(0, 3, 1, 1), crs=32606)
        correlation = VariogramModel("exponential", effective_range=2, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # The explicit eager approximation is the reference for the default chunked grid
        expected = raster.random_field(structure, random_state=4, backend="gpytorch", gpytorch_inducing_points=256)
        delegated = structure.generate_random_field(
            like=raster, random_state=4, backend="gpytorch", gpytorch_inducing_points=256
        )
        lazy = raster.random_field(structure, random_state=4, backend="gpytorch", chunksizes=(2, 3))
        assert hasattr(lazy.data, "compute")
        np.testing.assert_array_equal(delegated.to_nanarray(), expected.to_nanarray())
        np.testing.assert_allclose(np.asarray(lazy.compute()), expected.to_nanarray(), rtol=0, atol=1e-12)

    @pytest.mark.parametrize("backend", ["gstools", "gpytorch"])
    def test_random_field__raster_dask_mp_equal(self, tmp_path: Path, backend: Literal["gstools", "gpytorch"]) -> None:
        """Checks that both backends match eager fields across Dask and MP raster tiles."""

        pytest.importorskip(backend)

        # Seven columns leave a one-column edge tile; the masked pixel checks per-tile source reads
        values = np.ma.masked_array(np.ones((5, 7)), mask=np.eye(5, 7, dtype=bool))
        source = gu.Raster.from_array(values, transform=from_origin(0, 5, 1, 1), crs=32606, nodata=-99999)
        quality = gu.Raster.from_array(
            np.tile(np.linspace(0, 1, 7), (5, 1)), transform=source.transform, crs=source.crs
        )
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        correlation = VariogramModel("gaussian", effective_range=4, partial_sill=1)
        structure = gu.ErrorStructure(
            [gu.ErrorComponent("measurement", 0.5), gu.ErrorComponent("spatial", magnitude, correlation)]
        )
        inducing_points = 64 if backend == "gpytorch" else None
        expected = source.random_field(
            structure,
            predictors={"quality": quality},
            random_state=7,
            n_fields=2,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )

        # Read raster and predictor from files without loading either complete array
        source_path = tmp_path / "source.tif"
        quality_path = tmp_path / "quality.tif"
        source.to_file(source_path)
        quality.to_file(quality_path)
        chunked_source = gu.Raster(source_path)
        chunked_quality = gu.Raster(quality_path)
        lazy = chunked_source.random_field(
            structure,
            predictors={"quality": chunked_quality},
            random_state=7,
            n_fields=2,
            chunksizes=(3, 3),
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert not chunked_source.is_loaded and not chunked_quality.is_loaded
        assert all(hasattr(field.data, "compute") for field in lazy)

        # Worker tiles write one output per field; both inputs remain unloaded
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=(3, 3), outfile=str(tmp_path / "field.tif"), cluster=cluster)
            multiproc = chunked_source.random_field(
                structure,
                predictors={"quality": chunked_quality},
                random_state=7,
                n_fields=2,
                mp_config=config,
                backend=backend,
                gpytorch_inducing_points=inducing_points,
            )
        assert not chunked_source.is_loaded and not chunked_quality.is_loaded
        assert all(not field.is_loaded for field in multiproc)

        # Compare both realizations at each valid pixel, including tile boundaries
        for eager_field, lazy_field, mp_field in zip(expected, lazy, multiproc):
            computed = np.asarray(lazy_field.compute())
            saved = mp_field.to_nanarray()
            reference = eager_field.to_nanarray()
            if backend == "gstools":
                np.testing.assert_array_equal(computed, reference)
                np.testing.assert_array_equal(saved, reference)
            else:
                # Different chunk shapes can change the final interpolation sum's rounding
                np.testing.assert_allclose(computed, reference, rtol=0, atol=1e-12)
                np.testing.assert_allclose(saved, reference, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("backend", ["gstools", "gpytorch"])
    def test_random_field__point_dask_mp_equal(self, tmp_path: Path, backend: Literal["gstools", "gpytorch"]) -> None:
        """Checks that both backends match eager fields across Dask and MP point rows."""

        pytest.importorskip(backend)
        dask_geopandas = pytest.importorskip("dask_geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        _register_dask_pointcloud_accessor()

        # Nine irregular points split into 4/4/1 rows so the final partition tests row offsets
        points = gu.PointCloud.from_xyz(
            [0, 1, 3, 4, 6, 7, 9, 10, 12], [0, 2, 1, 3, 2, 5, 4, 6, 5], np.arange(9), crs=32631
        )
        points.ds["quality"] = np.linspace(0, 1, 9)
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        correlation = VariogramModel("exponential", effective_range=4, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", magnitude, correlation)])
        inducing_points = 64 if backend == "gpytorch" else None
        expected = points.random_field(
            structure,
            predictors={"quality": "quality"},
            random_state=11,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )

        # Existing Dask partitions and requested row chunks should both stay lazy
        dask_points = dask_geopandas.from_geopandas(points.ds, npartitions=3, sort=False)
        dask_points.pc.data_column = points.data_column
        lazy = dask_points.pc.random_field(
            structure,
            predictors={"quality": "quality"},
            random_state=11,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        rechunked = dask_points.pc.random_field(
            structure,
            predictors={"quality": "quality"},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        requested = points.random_field(
            structure,
            predictors={"quality": "quality"},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        array_lazy = points.random_field(
            structure,
            predictors={"quality": np.linspace(0, 1, 9)},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert hasattr(lazy, "compute") and hasattr(rechunked, "compute") and hasattr(requested, "compute")
        assert not lazy.pc.is_loaded and not rechunked.pc.is_loaded and not requested.pc.is_loaded
        assert not array_lazy.pc.is_loaded

        # Read worker rows from an unopened GeoPackage and write a new point file
        source_path = tmp_path / "points.gpkg"
        points.to_file(source_path)
        file_points = gu.PointCloud(source_path, data_column=points.data_column)
        file_lazy = file_points.random_field(
            structure,
            predictors={"quality": "quality"},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert not file_points.is_loaded and not file_lazy.pc.is_loaded
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=4, outfile=str(tmp_path / "field.gpkg"), cluster=cluster)
            multiproc = file_points.random_field(
                structure,
                predictors={"quality": "quality"},
                random_state=11,
                mp_config=config,
                backend=backend,
                gpytorch_inducing_points=inducing_points,
            )
        assert not file_points.is_loaded and not multiproc.is_loaded

        # Compare values in source order across all partition boundaries
        results = (lazy, rechunked, requested, array_lazy, file_lazy)
        for result in results:
            values = result.compute()[points.data_column].to_numpy()
            if backend == "gstools":
                np.testing.assert_array_equal(values, expected.data)
            else:
                # Different partition sizes can change the final interpolation sum's rounding
                np.testing.assert_allclose(values, expected.data, rtol=0, atol=1e-12)
        if backend == "gstools":
            np.testing.assert_array_equal(multiproc.data, expected.data)
        else:
            np.testing.assert_allclose(multiproc.data, expected.data, rtol=0, atol=1e-12)

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__point_geometry_z_chunks(self) -> None:
        """Checks that Dask point chunks write field values to geometry Z when no data column exists."""

        pytest.importorskip("dask_geopandas")

        # Seven 3D points leave a shorter final row partition
        x = np.arange(7, dtype=float)
        y = np.array([0, 1, 0, 2, 1, 3, 2], dtype=float)
        frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y, np.arange(7)), crs=32631)
        points = gu.PointCloud(frame, data_column=None)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        expected = points.random_field(structure, random_state=3)

        # Draw a lazy field, then compare point order and Z values with eager output
        lazy = points.random_field(structure, random_state=3, chunksizes=3)
        assert not lazy.pc.is_loaded
        result = lazy.compute()
        np.testing.assert_array_equal(result.geometry.x, expected.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, expected.geometry.y)
        np.testing.assert_array_equal(result.geometry.z, expected.geometry.z)


class TestRandomFieldErrors:
    """Test module for invalid random field options."""

    def test_random_field__error_invalid_backend(self) -> None:
        """Checks that random_field() rejects an unknown covariance backend before drawing a field."""

        # Single independent error component
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check error for unknown backend
        with pytest.raises(ValueError, match="backend"):
            gu.uncertainty.random_field(structure, coordinates=np.zeros((1, 2)), backend="unknown")  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_fields", [0, True])
    def test_random_field__error_invalid_field_count(self, n_fields: object) -> None:
        """Checks that random_field() requires a positive whole number of fields."""

        # Independent error with one source location
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="n_fields must be a positive integer"):
            gu.uncertainty.random_field(structure, coordinates=np.zeros((1, 2)), n_fields=n_fields)  # type: ignore[arg-type]

    def test_random_field__error_invalid_support(self) -> None:
        """Checks an error is raised when output locations are missing or supplied twice."""

        # Raster supplies its own pixel locations
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Reject repeated locations, absent support, and an invalid error model
        with pytest.raises(ValueError, match="coordinates must be omitted"):
            gu.uncertainty.random_field(structure, like=raster, coordinates=np.zeros((4, 2)))
        with pytest.raises(ValueError, match="Provide like or coordinates"):
            gu.uncertainty.random_field(structure)
        with pytest.raises(TypeError, match="error_structure must be"):
            gu.uncertainty.random_field(None, coordinates=np.zeros((1, 2)))  # type: ignore[arg-type]

    def test_random_field__error_invalid_predictors(self) -> None:
        """Checks that point predictor names exist and raster predictor arrays cover every output pixel."""

        # Point cloud and raster with one independent error component
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Reject missing table columns and arrays shorter than the output grid
        with pytest.raises(ValueError, match="predictor column 'slope' does not exist"):
            points.random_field(structure, predictors={"slope": "slope"})
        with pytest.raises(ValueError, match="must be scalar or match every output location"):
            raster.random_field(structure, predictors={"slope": [0.0, 1.0]})

    def test_random_field__error_chunked_support(self) -> None:
        """Checks that chunk sizes match the input type and inducing grids use GPyTorch."""

        # Two points and a 2 x 2 raster for unsupported chunked combinations
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        component = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check each requirement after binding the source error model
        with pytest.raises(ValueError, match="Point cloud chunk size"):
            gu.uncertainty.random_field(component, like=points, chunksizes=(1, 1))
        with pytest.raises(ValueError, match="gpytorch_inducing_points must be"):
            raster.random_field(component, chunksizes=(1, 1), backend="gpytorch", gpytorch_inducing_points=1)
