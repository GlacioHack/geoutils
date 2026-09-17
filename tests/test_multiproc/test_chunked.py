"""Tests for dividing arrays and georeferenced grids into chunks."""

from typing import Any

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from geoutils.multiproc.chunked import (
    ChunkedGeoGrid,
    GeoGrid,
    iter_chunk_slices,
    normalize_chunks,
)


class TestArrayChunks:
    """
    Test module for manipulating array chunks: accepted sizes, edge chunks and the order of the resulting slices.
    """

    @pytest.mark.parametrize(
        "chunks,expected",
        [
            (3, ((3, 3, 1), (3, 2))),
            ((3, 2), ((3, 3, 1), (2, 2, 1))),
            (((2, 5), (1, 3, 1)), ((2, 5), (1, 3, 1))),
        ],
    )
    def test_normalize_chunks__supported_forms(
        self,
        chunks: int | tuple[int, int] | tuple[tuple[int, ...], tuple[int, ...]],
        expected: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> None:
        """Checks that square, rectangular and explicit chunks cover the complete array shape."""

        # Try square, rectangular and fully specified chunks on the same array
        normalized = normalize_chunks(chunks, shape=(7, 5))
        assert normalized == expected

    def test_iter_chunk_slices__row_order_and_clipped_edges(self) -> None:
        """Checks that array slices follow row order and stop at the array edges."""

        # Divide a shape that is not evenly divisible by either chunk size
        slices = list(iter_chunk_slices(shape=(5, 7), chunks=(2, 3)))
        locations = [tuple((part.start, part.stop) for part in chunk) for chunk in slices]

        # Visit each column chunk before moving to the next row chunk, clipping the final slices to the shape
        expected = [
            ((0, 2), (0, 3)),
            ((0, 2), (3, 6)),
            ((0, 2), (6, 7)),
            ((2, 4), (0, 3)),
            ((2, 4), (3, 6)),
            ((2, 4), (6, 7)),
            ((4, 5), (0, 3)),
            ((4, 5), (3, 6)),
            ((4, 5), (6, 7)),
        ]
        assert locations == expected


class TestChunkedGeoGrid:
    """Test module for splitting a georeferenced grid into spatial blocks."""

    def test_chunked_geogrid__block_shapes_and_locations(self) -> None:
        """Checks that uneven blocks have the full grid's resolution and occupy their matching locations."""

        # Create a five-row, seven-column grid divided into six uneven blocks
        grid = GeoGrid(transform=from_origin(100, 200, 10, 20), shape=(5, 7), crs=CRS.from_epsg(32633))
        chunked = ChunkedGeoGrid(grid, chunks=((2, 3), (3, 3, 1)))
        blocks = chunked.get_blocks_as_geogrids()

        # Check block order, shapes, upper-left coordinates
        expected = [
            ((2, 3), (100, 200)),
            ((2, 3), (130, 200)),
            ((2, 1), (160, 200)),
            ((3, 3), (100, 160)),
            ((3, 3), (130, 160)),
            ((3, 1), (160, 160)),
        ]
        actual = [(block.shape, (block.transform.c, block.transform.f)) for block in blocks]
        assert actual == expected
        assert all(block.res == grid.res and block.crs == grid.crs for block in blocks)
        assert chunked.flat_block_index((1, 2)) == 5


class TestChunkedErrors:
    """Test module for validation errors raised by array chunk and georeferenced grid helpers."""

    @pytest.mark.parametrize(
        "chunks,error",
        [
            (0, ValueError),
            ((3, 0), ValueError),
            (((2, 5), (2, 2)), ValueError),
            ((2, (2, 3)), TypeError),
            ((2, 3, 4), ValueError),
        ],
    )
    def test_normalize_chunks__error_invalid_forms(self, chunks: Any, error: type[Exception]) -> None:
        """Checks that zero sizes, incorrect totals and extra axes all raise an error."""

        with pytest.raises(error):
            normalize_chunks(chunks, shape=(7, 5))

    def test_chunked_geogrid__error_invalid_block_position(self) -> None:
        """Checks that block positions outside a georeferenced grid raise an error."""

        # Create a grid with two row blocks and three column blocks
        grid = GeoGrid(transform=from_origin(100, 200, 10, 20), shape=(5, 7), crs=CRS.from_epsg(32633))
        chunked = ChunkedGeoGrid(grid, chunks=((2, 3), (3, 3, 1)))

        # Reject the first row position beyond the grid
        with pytest.raises(IndexError):
            chunked.flat_block_index((2, 0))
