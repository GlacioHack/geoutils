"""Test tiling tools for arrays and rasters."""

import numpy as np
import pytest

import geoutils as gu
from geoutils import examples
from geoutils.raster.tiling import _generate_tiling_grid


class TestTiling:

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")

    def test_subdivide_array(self) -> None:
        # Import optional scikit-image or skip test
        pytest.importorskip("skimage")

        test_shape = (6, 4)
        test_count = 4
        subdivision_grid = gu.raster.subdivide_array(test_shape, test_count)

        assert subdivision_grid.shape == test_shape
        assert np.unique(subdivision_grid).size == test_count

        assert np.unique(gu.raster.subdivide_array((3, 3), 3)).size == 3

        with pytest.raises(ValueError, match=r"Expected a 2D shape, got 1D shape.*"):
            gu.raster.subdivide_array((5,), 2)

        with pytest.raises(ValueError, match=r"Shape.*smaller than.*"):
            gu.raster.subdivide_array((5, 2), 15)

    @pytest.mark.parametrize("overlap", [0, 5])  # type: ignore
    def test_tiling(self, overlap: int) -> None:

        # Test with mock data
        tiling_grid_mock = _generate_tiling_grid(0, 0, 100, 100, 50, 50, overlap)
        if overlap == 0:
            expected_tiling = np.array([[[0, 50, 0, 50], [0, 50, 50, 100]], [[50, 100, 0, 50], [50, 100, 50, 100]]])
            assert np.array_equal(tiling_grid_mock, expected_tiling)
        elif overlap == 5:
            expected_tiling = np.array(
                [
                    [[0, 55, 0, 55], [0, 55, 45, 100]],
                    [[45, 100, 0, 55], [45, 100, 45, 100]],
                ]
            )
            assert np.array_equal(tiling_grid_mock, expected_tiling)

        tiling_grid_mock = _generate_tiling_grid(0, 0, 55, 55, 50, 50, overlap)
        if overlap == 0:
            expected_tiling = np.array([[[0, 50, 0, 50], [0, 50, 50, 55]], [[50, 55, 0, 50], [50, 55, 50, 55]]])
            assert np.array_equal(tiling_grid_mock, expected_tiling)
        elif overlap == 5:
            expected_tiling = np.array([[[0, 55, 0, 55]]])
            assert np.array_equal(tiling_grid_mock, expected_tiling)

        # Test with real data
        img = gu.Raster(self.landsat_b4_path)

        # Define tiling parameters
        row_split, col_split = 100, 100
        row_max, col_max = img.shape

        # Generate the tiling grid
        tiling_grid = _generate_tiling_grid(0, 0, row_max, col_max, row_split, col_split, overlap)

        # Calculate expected number of tiles
        nb_row_tiles = np.ceil(row_max / row_split).astype(int)
        nb_col_tiles = np.ceil(col_max / col_split).astype(int)

        if 0 < col_max % col_split <= overlap:
            nb_col_tiles = max(nb_col_tiles - 1, 1)
        if 0 < row_max % row_split <= overlap:
            nb_row_tiles = max(nb_row_tiles - 1, 1)

        # Check that the tiling grid has the expected shape
        assert tiling_grid.shape == (nb_row_tiles, nb_col_tiles, 4)

        # Check the boundaries of the first and last tile
        assert np.array_equal(
            tiling_grid[0, 0],
            np.array(
                [
                    0,
                    min(row_split + overlap, row_max),
                    0,
                    min(col_split + overlap, col_max),
                ]
            ),
        )
        assert np.array_equal(
            tiling_grid[-1, -1],
            np.array(
                [
                    (nb_row_tiles - 1) * row_split - overlap,
                    row_max,
                    (nb_col_tiles - 1) * col_split - overlap,
                    col_max,
                ]
            ),
        )

        # Check if overlap is consistent between tiles
        for row in range(nb_row_tiles - 1):
            assert tiling_grid[row + 1, 0, 0] == tiling_grid[row, 0, 1] - 2 * overlap

        for col in range(nb_col_tiles - 1):
            assert tiling_grid[0, col + 1, 2] == tiling_grid[0, col, 3] - 2 * overlap

    def test_tiling_overlap_errors(self) -> None:
        with pytest.raises(ValueError):
            _generate_tiling_grid(0, 0, 100, 100, 50, 50, -1)
        with pytest.raises(TypeError):
            _generate_tiling_grid(0, 0, 100, 100, 50, 50, 0.5)  # type: ignore
