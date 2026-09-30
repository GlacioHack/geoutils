"""Test nodata propagation definitions."""

from typing import cast

import numpy as np
import pytest

from geoutils.operators.nodata import (
    NodataChoice,
    _mask_grid_near_invalid_points,
    _nearest_source_validity,
    _nodata_spread_distance,
    _resolve_nodata_handling,
)


class TestNodataHandling:
    """Test module for nodata propagation definitions."""

    @pytest.mark.parametrize("order,rounded_up,rounded_down", [(0, 0, 0), (1, 1, 0), (3, 2, 1), (5, 3, 2)])
    def test_nodata_spread_distance__rounding_and_fixed_distance(
        self, order: int, rounded_up: int, rounded_down: int
    ) -> None:
        """Checks that half-order spread distance (using interpolation order) is rounded properly."""

        assert _nodata_spread_distance(order, "half_order_up") == rounded_up
        assert _nodata_spread_distance(order, "half_order_down") == rounded_down
        assert _nodata_spread_distance(order, 5) == 5

    @pytest.mark.parametrize(
        ("choice", "order", "expected"),
        [
            ("gdal", 1, ("gdal", None)),
            ("ignore", None, ("ignore", None)),
            ("propagate", None, ("propagate", None)),
            (0, None, ("ignore", 0)),
            (2, None, ("ignore", 2)),
            ("half_order_down", 3, ("ignore", 1)),
            ("half_order_up", 3, ("ignore", 2)),
        ],
    )
    def test_resolve_nodata_handling(
        self, choice: NodataChoice, order: int | None, expected: tuple[str, int | None]
    ) -> None:
        """
        Checks that each nodata handling behaves properly: separate "gdal/ignore/propagate" rules with spread
        distance.
        """

        # Check resolved as expected
        assert _resolve_nodata_handling(choice, order) == expected

    @pytest.mark.parametrize("choice,order", [(True, 1), (-1, 1), ("unknown", 1), ("half_order_up", None)])
    def test_resolve_nodata_handling__error(self, choice: str | int, order: int | None) -> None:
        """Checks invalid distances or rules raise an error."""

        with pytest.raises(ValueError):
            _resolve_nodata_handling(cast(NodataChoice, choice), order)


class TestPointNodataMasks:
    """Tests for nearest point nodata behaviour."""

    def test_nearest_source_validity(self) -> None:
        """Checks the edge case of two sources at equal distance: we use GDAL tie break rule."""

        # We define two nearest points
        points = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        valid = np.array([False, True, True])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Check tie-break: we select only the right point
        result = _nearest_source_validity(points, valid, targets)
        np.testing.assert_array_equal(result, [True, False])

    @pytest.mark.parametrize("radius", [0, 1])
    def test_mask_grid_near_invalid_points(self, radius: int) -> None:
        """Checks that mask expansion uses output pixel distances even when X/Y resolutions differ."""

        # Non-square pixels, with edge neighbors one pixel from center
        values = np.ones((3, 3))
        coordinates = (np.array([0.0, 2.0, 4.0]), np.array([0.0, 3.0, 6.0]))
        _mask_grid_near_invalid_points(values, np.array([[2.0, 3.0]]), coordinates, res_x=2, res_y=3, radius=radius)

        # Check center mask at radius zero, plus four edge neighbors at radius one
        expected_mask = np.zeros((3, 3), dtype=bool)
        expected_mask[1, 1] = True
        if radius == 1:
            expected_mask[1, :] = True
            expected_mask[:, 1] = True
        np.testing.assert_array_equal(np.isnan(values), expected_mask)
