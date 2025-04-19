from __future__ import annotations

from typing import Any
import logging
import pytest

import numpy as np
from cmath import isnan

from geoutils._typing import NDArrayNum
from geoutils import examples
import geoutils as gu

class TestStats:

    landsat_b4_path = examples.get_path("everest_landsat_b4")
    landsat_rgb_path = examples.get_path("everest_landsat_rgb")
    aster_dem_path = examples.get_path("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path, landsat_rgb_path])  # type: ignore
    def test_get_stats(self, example: str, caplog) -> None:
        raster = gu.Raster(example)

        expected_stats = [
            "Mean",
            "Median",
            "Max",
            "Min",
            "Sum",
            "Sum of squares",
            "90th percentile",
            "LE90",
            "NMAD",
            "RMSE",
            "Standard deviation",
            "Valid count",
            "Total count",
            "Percentage valid points",
        ]

        expected_stats_mask = [
            "Valid inlier count",
            "Total inlier count",
            "Percentage inlier points",
            "Percentage valid inlier points",
        ]

        stat_types = (int, float, np.integer, np.floating)

        # Full stats
        stats = raster.get_stats()
        for name in expected_stats:
            assert name in stats
            assert isinstance(stats.get(name), stat_types)

        # With mask
        inlier_mask = raster.get_mask()
        stats_masked = raster.get_stats(inlier_mask=inlier_mask)
        for name in expected_stats_mask:
            assert name in stats_masked
            assert isinstance(stats_masked.get(name), stat_types)
            stats_masked.pop(name)
        assert stats_masked == stats

        # Empty mask
        empty_mask = np.ones_like(inlier_mask)
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask)
        assert "Empty raster, returns Nan for all stats" in caplog.text
        for name in expected_stats + expected_stats_mask:
            print(name)
            print(stats_masked)
            assert np.isnan(stats_masked.get(name))

        # Single stat
        for name in expected_stats:
            stat = raster.get_stats(stats_name=name)
            assert np.isfinite(stat)

        # Callable
        def percentile_95(data: NDArrayNum) -> np.floating[Any]:
            if isinstance(data, np.ma.MaskedArray):
                data = data.compressed()
            return np.nanpercentile(data, 95)

        stat = raster.get_stats(stats_name=percentile_95)
        assert isinstance(stat, np.floating)

        # Selected stats and callable
        stats_name = ["mean", "max", "std", "percentile_95"]
        stats = raster.get_stats(stats_name=["mean", "max", "std", percentile_95])
        for name in stats_name:
            assert name in stats
            assert stats.get(name) is not None

        # non-existing stat
        with caplog.at_level(logging.WARNING):
            stat = raster.get_stats(stats_name="80 percentile")
            assert isnan(stat)
        assert "Statistic name '80 percentile' is not recognized" in caplog.text

