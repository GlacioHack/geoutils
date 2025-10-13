from __future__ import annotations

import logging
from cmath import isnan
from typing import Any

import numpy as np
import pytest

import geoutils as gu
from geoutils import examples
from geoutils._typing import NDArrayNum


class TestStats:
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
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
            "IQR",
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
        print(stats)

        print("test 1")
        for name in expected_stats:
            assert name in stats
            assert isinstance(stats.get(name), stat_types)

        # With mask (inlier=True)
        print("test 2")

        inlier_mask = ~raster.get_mask()
        stats_masked = raster.get_stats(inlier_mask=inlier_mask)
        for name in expected_stats_mask:
            assert name in stats_masked
            assert isinstance(stats_masked.get(name), stat_types)
            stats_masked.pop(name)
        assert stats_masked == stats

        # Empty mask (=False)
        print("test 3")

        empty_mask = np.zeros_like(inlier_mask)
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask)
        assert "Empty raster, returns Nan for all stats" in caplog.text
        for name in expected_stats + expected_stats_mask:
            assert np.isnan(stats_masked.get(name))

        # Single stat
        print("test 4")

        for name in expected_stats:
            print("#", name)
            stat = raster.get_stats(stats_name=name)
            assert np.isfinite(stat)

        # Callable
        print("test 5")
        print("test 5 a")

        def percentile_95(data: NDArrayNum) -> np.floating[Any]:
            if isinstance(data, np.ma.MaskedArray):
                data = data.compressed()
            return np.nanpercentile(data, 95)

        stat = raster.get_stats(stats_name=percentile_95)
        assert isinstance(stat, np.floating)

        print("test 5 b")

        # Selected stats and callable
        stats_name = ["mean", "max", "std", "percentile_95"]
        stats = raster.get_stats(stats_name=["mean", "max", "std", percentile_95])
        print(stats)
        for name in stats_name:
            print(name)
            assert name in stats
            assert stats.get(name) is not None

        print("test 6")

        # non-existing stat
        with caplog.at_level(logging.WARNING):
            stat = raster.get_stats(stats_name="80 percentile")
            assert isnan(stat)
        assert "Statistic name '80 percentile' is not recognized" in caplog.text

        print("test 7")

        # IQR (scipy) validation with numpy
        nan_arr = raster.get_nanarray()
        if nan_arr.ndim == 3:
            nan_arr = nan_arr[0, :, :]
        assert raster.get_stats(stats_name="iqr") == pytest.approx(
            np.nanpercentile(nan_arr, 75) - np.nanpercentile(nan_arr, 25)
        )
