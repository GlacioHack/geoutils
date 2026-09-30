"""Tests for random error fields over rasters, point clouds, and explicit coordinates."""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

import geoutils as gu
from geoutils.stats.variography import VariogramModel


class TestRandomFieldSpatialSupport:
    """Test module for random fields returned as rasters and point clouds."""

    def test_generate_random_field__keeps_raster_support_and_mask(self) -> None:
        """Checks that generated fields have the same shape, coordinates, and missing cells as the raster."""

        # Small raster with missing cell to check output mask
        mask = np.zeros((2, 3), dtype=bool)
        mask[0, 1] = True
        values = np.ma.masked_array(np.ones((2, 3)), mask=mask)
        raster = gu.Raster.from_array(values, transform=from_origin(10, 20, 2, 2), crs=32606)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Compare values, grid and mask across all three APIs
        first = gu.uncertainty.random_field(structure, like=raster, random_state=5)
        second = structure.generate_random_field(raster, random_state=5)
        method_result = raster.random_field(structure, random_state=5)
        assert isinstance(first, gu.Raster) and isinstance(second, gu.Raster) and isinstance(method_result, gu.Raster)
        assert first.shape == raster.shape and first.transform == raster.transform and first.crs == raster.crs
        np.testing.assert_array_equal(first.get_mask(), mask)
        np.testing.assert_array_equal(first.data, second.data)
        np.testing.assert_array_equal(first.data, method_result.data)

    def test_random_field__point_method_keeps_coordinates(self) -> None:
        """Checks that PointCloud.random_field() replaces values without moving observation coordinates."""

        # Irregular points with independent unit errors
        points = gu.PointCloud.from_xyz([0, 2, 1, 4], [1, 0, 3, 2], [10, 11, 12, 13], crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Compare point coordinates and values with function output
        result = points.random_field(structure, random_state=8)
        expected = gu.uncertainty.random_field(structure, like=points, random_state=8)
        np.testing.assert_array_equal(result.geometry.x, points.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, points.geometry.y)
        np.testing.assert_array_equal(result.data, expected.data)

    def test_random_field__point_method_uses_index_as_source_identity(self) -> None:
        """Checks that a labelled Gaussian field binds to point identities rather than row positions."""

        # Named points with fixed errors in reversed label order
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [10, 20], crs=32631)
        points.ds.index = pd.Index(["left", "right"])
        labels = pd.Index(["right", "left"])
        covariance = pd.DataFrame(np.zeros((2, 2)), index=labels, columns=labels)
        mean = pd.Series([7.0, 3.0], index=labels)
        structure = gu.ErrorStructure.from_gaussian(covariance, mean=mean)

        # Check errors matched by label, returned in point order
        result = points.random_field(structure, random_state=4)
        np.testing.assert_array_equal(result.data, [3.0, 7.0])

    def test_random_field__error_invalid_backend(self) -> None:
        """Checks that random_field() rejects an unknown covariance backend before drawing a field."""

        # Single independent error component
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check error for unknown backend
        with pytest.raises(ValueError, match="backend"):
            gu.uncertainty.random_field(structure, source_ids=[0], backend="unknown")  # type: ignore[arg-type]

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__correlated_support_is_reproducible(self) -> None:
        """Checks that a correlated component draws one reproducible field across its complete support."""

        # Exponential correlation across source coordinates
        coordinates = np.column_stack((np.arange(8, dtype=float), np.zeros(8)))
        correlation = VariogramModel("exponential", effective_range=4, partial_sill=1)
        structure = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])

        # Check field size and reproducibility with same seed
        first = gu.uncertainty.random_field(structure, source_ids=np.arange(8), coordinates=coordinates, random_state=3)
        second = gu.uncertainty.random_field(
            structure, source_ids=np.arange(8), coordinates=coordinates, random_state=3
        )
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
            source_ids=np.arange(4),
            coordinates=coordinates,
            random_state=9,
            backend="gpytorch",
        )
        second = gu.uncertainty.random_field(
            structure,
            source_ids=np.arange(4),
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
    """Test module for generating the same random field with different raster chunks."""

    @pytest.mark.skipif(find_spec("gstools") is None, reason="Requires GSTools")
    def test_random_field__raster_chunk_invariance(self) -> None:
        """Checks that changing raster chunks does not change independent or correlated field realizations."""

        # Independent + Gaussian components on chunked grid
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

        # Compare computed Dask field with eager result across chunk boundaries
        np.testing.assert_array_equal(np.asarray(lazy.compute()), expected.to_nanarray())
