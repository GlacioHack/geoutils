"""Tests for random error fields over rasters or point clouds."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.transform import from_origin, xy

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.raster.xr_accessor import open_raster
from geoutils.stats.variography import VariogramModel


class TestRandomField:
    """
    Test module for random fields.

    Tests for chunked backends (Dask/MP) are further below in TestRandomFieldChunked,
    while errors/warnings raised are in TestRandomFieldErrors.
    """

    def test_generate_random_field__match_raster(self) -> None:
        """Checks that the output random fields have grid as the input raster."""

        # We create a synthetic raster and error structure, with a masked data point
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
        """Checks that output random fields have the same coords as input point cloud."""

        # A synthetic point cloud and error structure
        points = gu.PointCloud.from_xyz([0, 2, 1, 4], [1, 0, 3, 2], [10, 11, 12, 13], crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check they match
        result = points.random_field(structure, random_state=8)
        expected = gu.uncertainty.random_field(structure, like=points, random_state=8)
        np.testing.assert_array_equal(result.geometry.x, points.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, points.geometry.y)
        np.testing.assert_array_equal(result.data, expected.data)

    @pytest.mark.parametrize("support", ["raster", "xarray", "pointcloud", "geodataframe"])
    @pytest.mark.parametrize("n_fields", [1, 2])
    def test_random_field__output_type(self, support: str, n_fields: int, tmp_path: Path) -> None:
        """Checks that each random field has the same spatial object type as the input."""

        # Build each public raster or point cloud input
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        raster_path = tmp_path / "source.tif"
        raster.to_file(raster_path)
        sources = {
            "raster": raster,
            "xarray": open_raster(str(raster_path)),
            "pointcloud": points,
            "geodataframe": points.ds,
        }
        source = sources[support]
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check one field and every member of a multi-field result through like=
        result = gu.uncertainty.random_field(structure, like=source, n_fields=n_fields, random_state=5)
        fields = result if isinstance(result, list) else [result]
        assert isinstance(result, list) == (n_fields > 1)
        assert len(fields) == n_fields
        assert all(type(field) is type(source) for field in fields)

    def test_random_field__duplicate_point(self) -> None:
        """Checks the edge case of duplicate points: should still have independent errors."""

        # We create a point cloud with twice the same point
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        points.ds.index = pd.Index(["same", "same"])
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check errors are independent
        result = points.random_field(structure, random_state=4)
        coordinates = np.column_stack((points.geometry.x.to_numpy(), points.geometry.y.to_numpy()))
        expected = structure.generate_random_field(coordinates=coordinates, random_state=4)
        np.testing.assert_array_equal(result.data, expected)
        assert result.data[0] != result.data[1]

    def test_random_field__raster_predictor(self) -> None:
        """Checks random field scaled by a raster predictor (without correlation yet)."""

        # We define a variable error magnitude with slope
        transform = from_origin(0, 2, 1, 1)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=transform, crs=32606)
        slope = gu.Raster.from_array(np.array([[0.0, 0.5], [1.0, 0.5]]), transform=transform, crs=32606)
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        variable = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        unit = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # With the same seed, slopes 0, 0.5, and 1 scale each unit-error draw by 1, 1.5, and 2
        result = raster.random_field(variable, predictors={"slope": slope}, random_state=5)
        reference = raster.random_field(unit, random_state=5)
        expected = reference.to_nanarray() * np.array([[1.0, 1.5], [2.0, 1.5]])
        np.testing.assert_allclose(result.to_nanarray(), expected)

    def test_random_field__scalar_predictor(self) -> None:
        """Checks that a scalar predictor value scales error magnitude everywhere (without correlation yet)."""

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

    def test_random_field__point_predictor(self) -> None:
        """Checks that a point cloud predictor scales properly error magnitude (without correlation yet)."""

        # We create a synthetic point cloud with variable error magnitude
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        points.ds["slope"] = [0.0, 1.0]
        statistics = pd.DataFrame({"nmad": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="slope"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        variable = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        unit = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # We check the scaling of the error is properly applied with predictors
        result = points.random_field(variable, predictors={"slope": "slope"}, random_state=5)
        reference = points.random_field(unit, random_state=5)
        np.testing.assert_allclose(result.data, reference.data * [1, 2])

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__single_axis(self) -> None:
        """Checks that a model using only X axis produces matching errors along each grid column."""

        # We define a correlation from a variogram with a single active dimension
        raster = gu.Raster.from_array(np.ones((2, 3)), transform=from_origin(0, 2, 1, 1), crs=32606)
        correlation = VariogramModel("gaussian", effective_range=3, partial_sill=1, active_dims=(0,))
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # We check the random field repeats across rows at the same X coords
        result = raster.random_field(structure, random_state=7)
        np.testing.assert_array_equal(result.to_nanarray()[0], result.to_nanarray()[1])

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__reproducible(self) -> None:
        """Checks that a correlated component is reproducible with a random seed."""

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
    def test_random_field__gpytorch_irregular(self) -> None:
        """Checks that a GPyTorch returns reproducible values on irregular points."""

        # Create irregular points with correlation model supported by GPyTorch
        coordinates = np.array([[0.0, 0.0], [0.5, 1.5], [2.0, 0.25], [3.0, 2.0]])
        correlation = VariogramModel("exponential", effective_range=3, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1.5, correlation)])

        # We check reproducibility with the same seed
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

        # A rotated raster gives GPyTorch an unstructured set of pixel centers
        transform = Affine(1, 0.3, 0, 0.2, -1, 2)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=transform, crs=32606)
        rows, columns = np.indices(raster.shape)
        x, y = xy(transform, rows, columns)
        raster_coordinates = np.column_stack((np.ravel(x), np.ravel(y)))

        # Compare the raster result with a draw at exactly the same coordinates
        raster_field = raster.random_field(structure, random_state=9, backend="gpytorch")
        coordinate_field = gu.uncertainty.random_field(
            structure, coordinates=raster_coordinates, random_state=9, backend="gpytorch"
        )
        np.testing.assert_array_equal(raster_field.to_nanarray().ravel(), coordinate_field)

    @pytest.mark.skipif(find_spec("gpytorch") is None or find_spec("torch") is None, reason="Requires GPyTorch")
    def test_random_field__gpytorch_regular(self) -> None:
        """Checks that GPyTorch return reproducible values on regular points."""

        # We create a small raster
        values = np.ma.masked_array(np.ones((3, 4)), mask=np.eye(3, 4, dtype=bool))
        raster = gu.Raster.from_array(values, transform=from_origin(10, 20, 2, 2), crs=32606)
        correlation = VariogramModel("exponential", effective_range=4, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # We check reproducible values with the same seed
        first = raster.random_field(structure, random_state=5, backend="gpytorch")
        second = raster.random_field(structure, random_state=5, backend="gpytorch")
        assert first.shape == raster.shape and first.transform == raster.transform and first.crs == raster.crs
        np.testing.assert_array_equal(first.get_mask(), raster.get_mask())
        np.testing.assert_array_equal(first.data, second.data)


class TestRandomFieldChunked:
    """Test module for random field with chunked (Dask/MP) backends."""

    @pytest.mark.parametrize("backend", ["gstools", "gpytorch"])
    def test_random_field__raster_dask_mp_equal(self, tmp_path: Path, backend: Literal["gstools", "gpytorch"]) -> None:
        """Checks that both Dask/MP backends match in-memory random fields."""

        pytest.importorskip(backend)

        # 1/ We create synthetic data, and run in-memory
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

        # 2/ We do the same with Dask/MP
        source_path = tmp_path / "source.tif"
        quality_path = tmp_path / "quality.tif"
        source.to_file(source_path)
        quality.to_file(quality_path)
        chunked_source = gu.Raster(source_path)
        chunked_quality = gu.Raster(quality_path)
        lazy_source = open_raster(str(source_path), chunks={"y": 3, "x": 3})
        lazy_quality = open_raster(str(quality_path), chunks={"y": 3, "x": 3})
        lazy = lazy_source.rst.random_field(
            structure,
            predictors={"quality": lazy_quality},
            random_state=7,
            n_fields=2,
            chunksizes=(3, 3),
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert not chunked_source.is_loaded and not chunked_quality.is_loaded
        assert all(hasattr(field, "rst") for field in lazy)
        assert hasattr(lazy_source.data, "compute") and hasattr(lazy_quality.data, "compute")
        assert all(hasattr(field.data, "compute") for field in lazy)
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

        # 3/ We check all are equal
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

        # 1/ We create synthetic data, and run in-memory
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

        # 2/ We do the same with Dask/MP
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
        requested = gu.uncertainty.random_field(
            structure,
            like=dask_points,
            predictors={"quality": "quality"},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        array_lazy = dask_points.pc.random_field(
            structure,
            predictors={"quality": np.linspace(0, 1, 9)},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert all(
            isinstance(result, dask_geopandas.GeoDataFrame) for result in (lazy, rechunked, requested, array_lazy)
        )
        assert hasattr(lazy, "compute") and hasattr(rechunked, "compute") and hasattr(requested, "compute")
        assert not lazy.pc.is_loaded and not rechunked.pc.is_loaded and not requested.pc.is_loaded
        assert not array_lazy.pc.is_loaded

        source_path = tmp_path / "points.gpkg"
        points.to_file(source_path)
        file_points = gu.PointCloud(source_path, data_column=points.data_column)
        from geoutils.pointcloud.pd_accessor import open_pointcloud

        file_dataframe = open_pointcloud(str(source_path), data_column=points.data_column, chunks=4)
        file_lazy = file_dataframe.pc.random_field(
            structure,
            predictors={"quality": "quality"},
            random_state=11,
            chunksizes=4,
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert isinstance(file_lazy, dask_geopandas.GeoDataFrame)
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
        assert isinstance(multiproc, gu.PointCloud)

        # 3/ We check all are equal
        results = (lazy, rechunked, requested, array_lazy, file_lazy)
        for result in results:
            values = result.compute()[points.data_column].to_numpy()
            if backend == "gstools":
                np.testing.assert_array_equal(values, expected.data)
            else:
                np.testing.assert_allclose(values, expected.data, rtol=0, atol=1e-12)
        if backend == "gstools":
            np.testing.assert_array_equal(multiproc.data, expected.data)
        else:
            np.testing.assert_allclose(multiproc.data, expected.data, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("backend", ["gstools", "gpytorch"])
    @pytest.mark.parametrize("active_dims", [None, (0,)])
    def test_random_field__rotated_raster(
        self, active_dims: tuple[int, ...] | None, backend: Literal["gstools", "gpytorch"], tmp_path: Path
    ) -> None:
        """Checks that rotated raster properly use chunk coordinates for the random field."""

        pytest.importorskip(backend)

        # We define a rotated raster
        transform = Affine(2, 0.5, 10, 0.25, -2, 20)
        raster = gu.Raster.from_array(np.ones((3, 4)), transform=transform, crs=32606)
        raster_path = tmp_path / "source.tif"
        raster.to_file(raster_path)
        source = open_raster(str(raster_path))
        correlation = VariogramModel("gaussian", effective_range=5, partial_sill=1, active_dims=active_dims)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # We run in-memory and chunked random fields
        assert not hasattr(source.data, "compute")
        inducing_points = 64 if backend == "gpytorch" else None
        expected = source.rst.random_field(
            structure, random_state=7, backend=backend, gpytorch_inducing_points=inducing_points
        )
        lazy = source.rst.random_field(
            structure,
            random_state=7,
            chunksizes=(2, 3),
            backend=backend,
            gpytorch_inducing_points=inducing_points,
        )
        assert hasattr(lazy, "rst")
        assert hasattr(lazy.data, "compute")
        assert lazy.data.chunks == ((2, 1), (3, 1))

        # Values should be almost equal
        np.testing.assert_allclose(np.asarray(lazy.compute()), np.asarray(expected), rtol=0, atol=1e-12)

    @pytest.mark.skipif(find_spec("gpytorch") is None, reason="Requires GPyTorch")
    def test_random_field__gpytorch_default(self, tmp_path: Path) -> None:
        """Checks that a chunked GPyTorch field uses its default inducing grid."""

        # We create a small raster
        raster = gu.Raster.from_array(np.ones((3, 4)), transform=from_origin(0, 3, 1, 1), crs=32606)
        raster_path = tmp_path / "source.tif"
        raster.to_file(raster_path)
        source = open_raster(str(raster_path))
        correlation = VariogramModel("exponential", effective_range=2, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 1, correlation)])

        # We should have exact equality of all 3 because of the default inducing grid
        expected = source.rst.random_field(structure, random_state=4, backend="gpytorch", gpytorch_inducing_points=256)
        delegated = structure.generate_random_field(
            like=source, random_state=4, backend="gpytorch", gpytorch_inducing_points=256
        )
        lazy = source.rst.random_field(structure, random_state=4, backend="gpytorch", chunksizes=(2, 3))
        assert hasattr(lazy, "rst")
        assert hasattr(lazy.data, "compute")
        np.testing.assert_array_equal(np.asarray(delegated), np.asarray(expected))
        np.testing.assert_allclose(np.asarray(lazy.compute()), np.asarray(expected), rtol=0, atol=1e-12)

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__point_default(self, tmp_path: Path) -> None:
        """Checks that Dask point chunks write field values to geometry Z when no data column exists."""

        # Create a synthetic point cloud
        x = np.arange(7, dtype=float)
        y = np.array([0, 1, 0, 2, 1, 3, 2], dtype=float)
        frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y, np.arange(7)), crs=32631)
        points = gu.PointCloud(frame, data_column=None)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        # Get in-memory random field
        expected = points.random_field(structure, random_state=3)

        # Same with Dask
        pytest.importorskip("dask_geopandas")
        from geoutils.pointcloud.pd_accessor import open_pointcloud

        source_path = tmp_path / "points.gpkg"
        points.to_file(source_path)
        dask_points = open_pointcloud(str(source_path), data_column=None, chunks=3)
        lazy = dask_points.pc.random_field(structure, random_state=3, chunksizes=3)
        assert not lazy.pc.is_loaded
        result = lazy.compute()

        # Check 3D points are the same
        np.testing.assert_array_equal(result.geometry.x, expected.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, expected.geometry.y)
        np.testing.assert_array_equal(result.geometry.z, expected.geometry.z)


class TestRandomFieldErrors:
    """Test module for errors/warnings of random fields."""

    def test_random_field__error_invalid_backend(self) -> None:
        """Checks an invalid backend name."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="backend"):
            gu.uncertainty.random_field(structure, coordinates=np.zeros((1, 2)), backend="unknown")  # type: ignore[arg-type]

    @pytest.mark.parametrize("n_fields", [0, True])
    def test_random_field__error_invalid_nfields(self, n_fields: object) -> None:
        """Checks a valid number of realizations/fields is passed."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="n_fields must be a positive integer"):
            gu.uncertainty.random_field(structure, coordinates=np.zeros((1, 2)), n_fields=n_fields)  # type: ignore[arg-type]

    def test_random_field__error_invalid_support(self) -> None:
        """Checks output locations exist properly."""

        # Synthetic raster
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Error on raster + coords at once
        with pytest.raises(ValueError, match="coordinates must be omitted"):
            gu.uncertainty.random_field(structure, like=raster, coordinates=np.zeros((4, 2)))
        # Error on absent support
        with pytest.raises(ValueError, match="Provide like or coordinates"):
            gu.uncertainty.random_field(structure)
        # Error on absent error structure
        with pytest.raises(TypeError, match="error_structure must be"):
            gu.uncertainty.random_field(None, coordinates=np.zeros((1, 2)))  # type: ignore[arg-type]

    def test_random_field__error_invalid_predictors(self) -> None:
        """Checks that predictor names exist and have right shape."""

        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Error on non-existing predicgtor
        with pytest.raises(ValueError, match="predictor column 'slope' does not exist"):
            points.random_field(structure, predictors={"slope": "slope"})
        # Error on wrong shape
        with pytest.raises(ValueError, match="must be scalar or match every output location"):
            raster.random_field(structure, predictors={"slope": [0.0, 1.0]})

    def test_random_field__error_chunked_support(self) -> None:
        """Checks that chunk sizes match input (raster = 1/2D, point = 1D), and inducing grid defined."""

        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        component = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Error on 2D chunks for points
        with pytest.raises(ValueError, match="Point cloud chunk size"):
            gu.uncertainty.random_field(component, like=points, chunksizes=(1, 1))
        # Error on wrong inducing point size
        with pytest.raises(ValueError, match="gpytorch_inducing_points must be"):
            raster.random_field(component, chunksizes=(1, 1), backend="gpytorch", gpytorch_inducing_points=1)

    def test_random_field__error_input_type(self) -> None:
        """Checks an error is raised for mismatched input/output types."""

        # In-memory objects cannot return a Dask/MP object
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Error on a wrong point input, from PointCloud or eager DataFrame
        with pytest.raises(ValueError, match="Dask point chunks require a Dask GeoDataFrame input"):
            points.random_field(structure, chunksizes=1)
        with pytest.raises(ValueError, match="Dask point chunks require a Dask GeoDataFrame input"):
            points.ds.pc.random_field(structure, chunksizes=1)

    def test_random_field__error_multiproc_xarray(self, tmp_path: Path) -> None:
        """Checks an error is raised for MP with Xarray input."""

        # Pass a Multiproc object to an Xarray input
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32606)
        raster_path = tmp_path / "source.tif"
        raster.to_file(raster_path)
        source = open_raster(str(raster_path))
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        config = MultiprocConfig(chunks=(1, 1), outfile=str(tmp_path / "field.tif"))

        # Error raised
        with pytest.raises(ValueError, match="Multiprocessing raster fields require a Raster input"):
            source.rst.random_field(structure, mp_config=config)

    def test_random_field__error_multiproc_dask_points(self, tmp_path: Path) -> None:
        """Checks an error is raised for MP on Dask input."""

        dask_geopandas = pytest.importorskip("dask_geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # We create a Dask point input
        _register_dask_pointcloud_accessor()
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        dask_points = dask_geopandas.from_geopandas(points.ds, npartitions=2, sort=False)
        dask_points.pc.data_column = points.data_column
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        config = MultiprocConfig(chunks=1, outfile=str(tmp_path / "field.gpkg"))

        # Error when passing MP config
        with pytest.raises(ValueError, match="Multiprocessing point fields require a PointCloud or GeoDataFrame input"):
            dask_points.pc.random_field(structure, mp_config=config)
