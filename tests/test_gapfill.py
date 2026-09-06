"""Test methods that fill missing raster values from surrounding cells."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio as rio

from geoutils import Raster


class TestGapFill:
    """Test interpolation of nodata cells from nearby valid raster values."""

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
            array,
            transform=rio.transform.from_origin(0, 1, 1, 1),
            crs=4326,
            nodata=-9999,
        )

        result = raster.fill_nodata(max_search_distance=2, interpolation=interpolation)  # type: ignore[arg-type]
        assert np.array_equal(result.get_nanarray(), np.array([[1.0, 1.0, 1.0, np.nan]]), equal_nan=True)
        assert np.array_equal(raster.get_nanarray(), np.array([[1.0, np.nan, np.nan, np.nan]]), equal_nan=True)
