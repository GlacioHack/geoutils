"""Test exact polygon intersections with cells of a regular grid."""

from importlib.util import find_spec
from typing import Literal

import numpy as np
import pytest
import rasterio as rio
import shapely
from affine import Affine

from geoutils.operators.overlap import (
    GridIntersection,
    _grid_intersection_fractions,
    _grid_intersection_fractions_from_corners,
)


class TestGridIntersection:
    """Test module for grouped intersection arrays, empty geometries and invalid cell coverage."""

    def test_gridinters__for_geometry(self) -> None:
        """Checks that for_geometry() returns proper empty/fraction coverages."""

        # Define an empty first geometry followed by two covered cells
        # Note: consecutive offsets define slices for rows, columns and fractions, so: 0:0 (empty) and 0:2 (two cells)
        overlap = GridIntersection(
            offsets=np.array([0, 0, 2]),
            rows=np.array([1, 1]),
            columns=np.array([0, 1]),
            fractions=np.array([0.25, 0.75]),
        )
        empty = overlap.for_geometry(0)
        rows, columns, fractions = overlap.for_geometry(1)

        # Check empty group and following cell indices/fractions
        assert overlap.geometry_count == 2
        assert all(len(array) == 0 for array in empty)
        np.testing.assert_array_equal(rows, [1, 1])
        np.testing.assert_array_equal(columns, [0, 1])
        np.testing.assert_array_equal(fractions, [0.25, 0.75])

    @pytest.mark.parametrize("index,error", [(True, TypeError), (1.5, TypeError), (-1, IndexError), (1, IndexError)])
    def test_for_geometry__error_index(self, index: int | float, error: type[Exception]) -> None:
        """Checks that geometry selection rejects noninteger indexes and positions outside the stored groups."""

        overlap = GridIntersection(np.array([0, 1]), np.array([0]), np.array([0]), np.array([1.0]))
        with pytest.raises(error):
            overlap.for_geometry(index)  # type: ignore[arg-type]

    @pytest.mark.parametrize("fraction", [0, -0.5, 1.5, np.nan, np.inf])
    def test_grid_intersection__error_fraction(self, fraction: float) -> None:
        """Checks that stored cell coverage is finite, positive and no greater than a complete cell."""

        # Reject zero-area contacts as well as invalid coverage fractions
        with pytest.raises(ValueError, match="coverage|fractions"):
            GridIntersection(np.array([0, 1]), np.array([0]), np.array([0]), np.array([fraction]))


class TestGridIntersectionFractions:
    """Test module for the cells and covered fractions returned by polygon intersections."""

    def test_grid_intersection_fractions__partial_cells(self) -> None:
        """Checks that a polygon centered on a corner covers a quarter of the 4 neighboring cells."""

        # Unit square raster centered on a corner shared by 4 cells
        transform = rio.transform.from_origin(0, 2, 1, 1)
        geometry = shapely.box(0.5, 0.5, 1.5, 1.5)
        corners = np.asarray(geometry.exterior.coords)[:-1][None, ...]

        # Intersect with Shapely
        overlap = _grid_intersection_fractions([geometry], transform, (2, 2))
        aligned_overlap = _grid_intersection_fractions_from_corners(corners, transform, (2, 2))
        rows, columns, fractions = overlap.for_geometry(0)

        # Check quarter(=0.25) coverage and ordering
        np.testing.assert_array_equal(rows, [0, 0, 1, 1])
        np.testing.assert_array_equal(columns, [0, 1, 0, 1])
        np.testing.assert_allclose(fractions, 0.25)
        np.testing.assert_array_equal(aligned_overlap.rows, overlap.rows)
        np.testing.assert_array_equal(aligned_overlap.columns, overlap.columns)
        np.testing.assert_allclose(aligned_overlap.fractions, overlap.fractions)

    def test_grid_intersection_fractions__rotated_grid(self) -> None:
        """Checks that intersection fractions use affine cell polygons for a rotated, non-square grid."""

        # Rotated grid with cell area six
        transform = Affine.translation(100, 200) * Affine.rotation(23) * Affine.scale(2, -3)
        columns = np.array([1, 2, 2, 1])
        rows = np.array([1, 1, 2, 2])
        x = transform.a * columns + transform.b * rows + transform.c
        y = transform.d * columns + transform.e * rows + transform.f
        geometry = shapely.Polygon(np.column_stack((x, y)))

        # Intersect polygon following one rotated cell
        overlap = _grid_intersection_fractions_from_corners(
            np.asarray(geometry.exterior.coords)[:-1][None, ...],
            transform,
            (4, 4),
        )
        selected_rows, selected_columns, fractions = overlap.for_geometry(0)

        # Check single fully covered cell
        np.testing.assert_array_equal(selected_rows, [1])
        np.testing.assert_array_equal(selected_columns, [1])
        assert np.allclose(fractions, [1.0], equal_nan=True)

    def test_grid_intersection_fractions__unaligned_corners(self) -> None:
        """Checks that unaligned quadrilateral corners fall back to the same exact Shapely intersections."""

        # Diamond crossing cells of a 3 x 3 grid
        transform = rio.transform.from_origin(0, 3, 1, 1)
        corners = np.array([[[1.5, 2.8], [2.8, 1.5], [1.5, 0.2], [0.2, 1.5]]])
        geometry = shapely.Polygon(corners[0])

        # Compare corner calculation with Shapely reference
        expected = _grid_intersection_fractions([geometry], transform, (3, 3))
        actual = _grid_intersection_fractions_from_corners(corners, transform, (3, 3))

        # Check matching cells and coverage fractions
        np.testing.assert_array_equal(actual.offsets, expected.offsets)
        np.testing.assert_array_equal(actual.rows, expected.rows)
        np.testing.assert_array_equal(actual.columns, expected.columns)
        np.testing.assert_allclose(actual.fractions, expected.fractions)

    @pytest.mark.parametrize("angle", [11, 29])
    @pytest.mark.parametrize("backend", ["auto", "numba", "exactextract"])
    def test_grid_intersection_fractions__rotated_quadrilaterals(
        self, angle: int, backend: Literal["auto", "numba", "exactextract"]
    ) -> None:
        """Checks that every raster overlap method matches Shapely on a rotated grid."""

        if backend in ("numba", "exactextract"):
            pytest.importorskip(backend)

        # Large coordinates expose cancellation while unequal cell widths give several partial overlaps
        transform = Affine.translation(500_000, 5_100_000) * Affine.rotation(angle) * Affine.scale(100, -150)
        pixel_corners = np.array(
            [
                [[1.2, 2.1], [3.4, 2.3], [3.1, 4.2], [1.0, 3.7]],
                [[3.5, 3.0], [5.4, 2.7], [5.2, 4.8], [3.2, 5.1]],
            ]
        )
        x = transform.a * pixel_corners[..., 0] + transform.b * pixel_corners[..., 1] + transform.c
        y = transform.d * pixel_corners[..., 0] + transform.e * pixel_corners[..., 1] + transform.f
        corners = np.stack((x, y), axis=-1)

        # Compare each polygon's covered cells and fractions after sorting the backend's cell order
        expected = _grid_intersection_fractions_from_corners(corners, transform, (7, 7), backend="shapely")
        actual = _grid_intersection_fractions_from_corners(corners, transform, (7, 7), backend=backend)
        np.testing.assert_array_equal(actual.offsets, expected.offsets)
        for geometry_index in range(len(corners)):
            expected_rows, expected_columns, expected_fractions = expected.for_geometry(geometry_index)
            actual_rows, actual_columns, actual_fractions = actual.for_geometry(geometry_index)
            expected_ids = expected_rows * 7 + expected_columns
            actual_ids = actual_rows * 7 + actual_columns
            expected_order = np.argsort(expected_ids)
            actual_order = np.argsort(actual_ids)
            np.testing.assert_array_equal(actual_ids[actual_order], expected_ids[expected_order])
            assert np.allclose(actual_fractions[actual_order], expected_fractions[expected_order])

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_grid_intersection_fractions__error_missing_numba(self) -> None:
        """Checks an error is raised for an explicit Numba backend when Numba is unavailable."""

        # Unaligned footprint reaches the optional Numba overlap method
        transform = rio.transform.from_origin(0, 2, 1, 1)
        corners = np.array([[[0.2, 1.8], [1.8, 1.6], [1.6, 0.2], [0.2, 0.4]]])

        # Explicit selection must report the missing dependency instead of silently changing methods
        with pytest.raises(ImportError, match="Numba overlap requires numba"):
            _grid_intersection_fractions_from_corners(corners, transform, (2, 2), backend="numba")


class TestExactExtract:
    """Test module for ExactExtract coverage, agreement with Shapely and unsupported grid transforms."""

    def test_grid_intersection_fractions__exactextract_matches_shapely(self) -> None:
        """Checks that optional ExactExtract returns the same cells and covered fractions as Shapely."""

        pytest.importorskip("exactextract")

        # Slanted polygons with partial coverage and a shared cell
        transform = rio.transform.from_origin(0, 4, 1, 1)
        geometries = [
            shapely.Polygon([(0.2, 0.3), (2.7, 0.6), (2.4, 3.3), (0.4, 2.8)]),
            shapely.Polygon([(1.1, 1.2), (3.6, 1.4), (3.1, 3.8), (1.5, 3.4)]),
        ]

        # Compare per geometry after sorting cells (backend order may differ)
        shapely_overlap = _grid_intersection_fractions(geometries, transform, (4, 4), backend="shapely")
        exactextract_overlap = _grid_intersection_fractions(geometries, transform, (4, 4), backend="exactextract")
        for geometry_index in range(len(geometries)):
            expected_rows, expected_columns, expected_fractions = shapely_overlap.for_geometry(geometry_index)
            rows, columns, fractions = exactextract_overlap.for_geometry(geometry_index)
            expected_ids = expected_rows * 4 + expected_columns
            actual_ids = rows * 4 + columns
            actual_order = np.argsort(actual_ids)
            expected_order = np.argsort(expected_ids)
            np.testing.assert_array_equal(actual_ids[actual_order], expected_ids[expected_order])
            np.testing.assert_allclose(
                fractions[actual_order],
                expected_fractions[expected_order],
                rtol=5e-7,
                atol=5e-7,
            )

    def test_grid_intersection_fractions__error_exactextract_rotated_source_grid(self) -> None:
        """Checks that selecting ExactExtract rejects grids it cannot represent without resampling."""

        pytest.importorskip("exactextract")

        # Check ExactExtract rejection of rotated grid
        transform = Affine.translation(0, 4) * Affine.rotation(10) * Affine.scale(1, -1)
        geometry = shapely.box(0, 0, 2, 2)
        with pytest.raises(ValueError, match="north-up grid"):
            _grid_intersection_fractions([geometry], transform, (4, 4), backend="exactextract")

        # Check automatic Shapely selection for rotated grid
        automatic = _grid_intersection_fractions([geometry], transform, (4, 4), backend="auto")
        shapely_result = _grid_intersection_fractions([geometry], transform, (4, 4), backend="shapely")
        np.testing.assert_array_equal(automatic.rows, shapely_result.rows)
        np.testing.assert_array_equal(automatic.columns, shapely_result.columns)
        np.testing.assert_allclose(automatic.fractions, shapely_result.fractions)
