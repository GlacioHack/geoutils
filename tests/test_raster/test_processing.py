"""Test GDAL-style raster processing and metadata editing methods."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio as rio

from geoutils import Raster


class TestRasterProcessing:
    """Check connected-region cleanup and interpolation of nodata cells."""

    def test_sieve(self) -> None:
        """Remove an isolated category while retaining a larger connected region."""

        # One category has a single cell while the other has two connected cells
        array = np.ones((5, 5), dtype=np.uint8)
        array[1, 1] = 2
        array[3, 3:5] = 3
        raster = Raster.from_array(array, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)

        # GDAL replaces only the region below the two-pixel threshold
        result = raster.sieve(size=2)
        expected = array.copy()
        expected[1, 1] = 1
        assert np.array_equal(result.data, expected)
        assert np.array_equal(raster.data, array)

        # Cells excluded by a processing mask keep their original values
        excluded = np.ones(array.shape, dtype=bool)
        excluded[1, 1] = False
        masked_result = raster.sieve(size=2, mask=excluded)
        assert np.array_equal(masked_result.data, array)

    def test_sieve_errors(self) -> None:
        """Reject unsupported continuous values and invalid region definitions."""

        raster = Raster.from_array(
            np.ones((3, 3), dtype=np.float32), transform=rio.transform.from_origin(0, 3, 1, 1), crs=4326
        )
        with pytest.raises(ValueError, match="integer or Boolean"):
            raster.sieve(size=2)
        with pytest.raises(ValueError, match="strictly positive integer"):
            raster.astype(np.uint8).sieve(size=0)
        with pytest.raises(ValueError, match="connectivity.*4 or 8"):
            raster.astype(np.uint8).sieve(size=2, connectivity=6)  # type: ignore[arg-type]

    @pytest.mark.parametrize("interpolation", ["inv_dist", "nearest"])
    def test_fill_nodata(self, interpolation: str) -> None:
        """Fill reachable cells and retain nodata beyond the requested distance."""

        # A single valid edge cell makes the maximum distance easy to verify
        array = np.ma.array(
            [[1.0, -9999.0, -9999.0, -9999.0]],
            mask=[[False, True, True, True]],
            fill_value=-9999,
            dtype=np.float32,
        )
        raster = Raster.from_array(
            array, transform=rio.transform.from_origin(0, 1, 1, 1), crs=4326, nodata=-9999
        )

        result = raster.fill_nodata(max_search_distance=2, interpolation=interpolation)  # type: ignore[arg-type]
        assert np.array_equal(result.get_nanarray(), np.array([[1.0, 1.0, 1.0, np.nan]]), equal_nan=True)
        assert np.array_equal(raster.get_nanarray(), np.array([[1.0, np.nan, np.nan, np.nan]]), equal_nan=True)


class TestRasterEdit:
    """Check grouped metadata changes without modifying source raster data or metadata."""

    def test_edit(self) -> None:
        """Apply core GDAL-style metadata edits to a shallow raster copy."""

        transform = rio.transform.from_origin(0, 2, 1, 1)
        raster = Raster.from_array(
            np.ones((2, 2), dtype=np.int16),
            transform=transform,
            crs=4326,
            nodata=-9999,
            tags={"source": "test"},
            area_or_point="Area",
        )
        new_transform = rio.transform.from_origin(10, 20, 2, 2)

        # Omitted metadata is retained while tags are merged with the existing values
        edited = raster.edit(
            crs=32631,
            transform=new_transform,
            nodata=-32768,
            tags={"edited": "true"},
            area_or_point="Point",
        )
        assert edited.crs == rio.crs.CRS.from_epsg(32631)
        assert edited.transform == new_transform
        assert edited.nodata == -32768
        assert edited.tags["source"] == "test"
        assert edited.tags["edited"] == "true"
        assert edited.area_or_point == "Point"

        # Editing metadata does not alter the source raster
        assert raster.crs == rio.crs.CRS.from_epsg(4326)
        assert raster.transform == transform
        assert raster.nodata == -9999
        assert raster.tags == {"source": "test", "AREA_OR_POINT": "Area"}
        assert np.array_equal(edited.data, raster.data)

    def test_edit_clear_metadata(self) -> None:
        """Distinguish omitted arguments from metadata explicitly cleared with None."""

        raster = Raster.from_array(
            np.ones((2, 2), dtype=np.float32),
            transform=rio.transform.from_origin(0, 2, 1, 1),
            crs=4326,
            nodata=-9999,
            tags={"source": "test"},
        )
        edited = raster.edit(crs=None, nodata=None, tags=None, area_or_point=None)
        assert edited.crs is None
        assert edited.nodata is None
        assert edited.tags == {}
        assert edited.area_or_point is None
