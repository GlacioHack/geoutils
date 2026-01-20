from __future__ import annotations

import logging
from cmath import isnan
from typing import Any

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples
from geoutils._typing import NDArrayNum

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
]

expected_stats_count = [
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


def compare_dict(dict1: dict, dict2: dict) -> None:  # type: ignore
    assert len(dict1.keys()) == len(dict1.keys())
    for key in dict1.keys():
        assert key in dict2
        print("key", key)
        if dict1[key] is not np.nan:
            assert dict2[key] == pytest.approx(dict1[key], abs=1e-10)
        else:
            assert dict2[key] is np.nan


class TestStats:
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_rgb_path, aster_dem_path])  # type: ignore
    def test_get_stats_raster(self, example: str, caplog) -> None:
        """
        Verify get_stats() method for a raster, especially output stats for different inputs
        parameters and some stats.
        """
        raster = gu.Raster(example)

        # Full stats
        stats = raster.get_stats()
        assert len(stats) == len(expected_stats + expected_stats_count)
        for name in expected_stats + expected_stats_count:
            assert name in stats
            assert isinstance(stats.get(name), stat_types)

        # With mask (inlier=True)
        inlier_mask = ~raster.get_mask()
        stats_masked = raster.get_stats(inlier_mask=inlier_mask)
        assert len(stats_masked) == len(expected_stats + expected_stats_count + expected_stats_mask)
        for name in expected_stats_mask:
            assert name in stats_masked
            stats_masked.pop(name)
        assert stats_masked == stats

        # Empty mask (=False)
        empty_mask = np.zeros_like(inlier_mask)
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask)
        assert "Empty raster, returns Nan for all stats" in caplog.text
        assert len(stats_masked) == len(expected_stats + expected_stats_count + expected_stats_mask)
        for name in expected_stats:
            assert np.isnan(stats_masked.get(name))

        print('stats_masked.get("Valid count")', stats_masked.get("Valid count"))
        assert stats_masked.get("Valid count") == stats.get("Valid count")
        assert stats_masked.get("Total count") == stats.get("Total count")
        assert stats_masked.get("Percentage valid points") == stats.get("Percentage valid points")

        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask, stats_name="mean")
        assert np.isnan(stats_masked)
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask, stats_name="valid_count")
        assert stats_masked == stats.get("Valid count")
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask, stats_name="Valid inlier count")
        assert stats_masked == 0
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.get_stats(inlier_mask=empty_mask, stats_name="validinliercount")
        assert stats_masked == 0

        # Empty DEM
        dem_empty = gu.Raster.from_array(
            np.random.randint(42, size=(0, 0), dtype="uint8"),
            transform=rio.transform.from_origin(10, 20, 1, 1),
            crs=4326,
        )
        stats_empty = dem_empty.get_stats()
        assert len(stats_empty) == len(expected_stats + expected_stats_count)
        for name in expected_stats:
            assert np.isnan(stats_empty.get(name))
        assert stats_empty.get("Valid count") == 0
        assert stats_empty.get("Total count") == 0
        assert isnan(stats_empty.get("Percentage valid points"))

        # Single stat
        for name in expected_stats + expected_stats_count:
            stat = raster.get_stats(stats_name=name)
            assert np.isfinite(stat)

        # Alias stat
        assert raster.get_stats(stats_name="Valid count") == raster.get_stats(stats_name="valid_count")

        # Callable
        def percentile_95(data: NDArrayNum) -> np.floating[Any]:
            if isinstance(data, np.ma.MaskedArray):
                data = data.compressed()
            return np.nanpercentile(data, 95)

        stat = raster.get_stats(stats_name=percentile_95)
        assert isinstance(stat, np.floating)

        # Selected stats and callable
        stats_name = ["mean", "max", "std", "validinliercount", "percentile_95"]
        stats = raster.get_stats(stats_name=["mean", "max", "std", "validinliercount", percentile_95])
        assert len(stats_name) == len(stats_name)
        for name in stats_name:
            assert name in stats
            assert stats.get(name) is not None

        # Non-existing stats
        with caplog.at_level(logging.WARNING):
            stat = raster.get_stats(stats_name="80 percentile")
            assert isnan(stat)
        assert "Statistic name '80 percentile' is not recognized" in caplog.text

        with caplog.at_level(logging.WARNING):
            stat = raster.get_stats(stats_name=42)
            assert stat is None
        # assert "Statistic name '42' is not recognized string" in caplog.text

        # IQR (scipy) validation with numpy
        nan_arr = raster.get_nanarray()
        if nan_arr.ndim == 3:
            nan_arr = nan_arr[0, :, :]
        assert raster.get_stats(stats_name="iqr") == pytest.approx(
            np.nanpercentile(nan_arr, 75) - np.nanpercentile(nan_arr, 25)
        )

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_get_stats_raster_pointcloud(self, example: str, caplog) -> None:
        """
        Verify get_stats() method for a raster converted to pointcloud, especially output stats for different inputs
        parameters.
        """
        raster = gu.Raster(example)

        # Full stats
        stats = raster.to_pointcloud().get_stats()
        assert len(stats) == len(expected_stats + expected_stats_count)
        for name in expected_stats + expected_stats_count:
            assert name in stats
            assert isinstance(stats.get(name), stat_types)

        # Single stat
        for name in expected_stats + expected_stats_count:
            stat = raster.to_pointcloud().get_stats(stats_name=name)
            assert np.isfinite(stat)

        # Callable
        def percentile_95(data: NDArrayNum) -> np.floating[Any]:
            if isinstance(data, np.ma.MaskedArray):
                data = data.compressed()
            return np.nanpercentile(data, 95)

        # Selected stats and callable
        stats_name = ["mean", "max", "std", "percentile_95"]
        stats = raster.to_pointcloud().get_stats(stats_name=["mean", "max", "std", percentile_95])
        assert len(stats) == len(stats_name)
        for name in stats_name:
            assert name in stats
            assert stats.get(name) is not None

        # Non-existing stats
        with caplog.at_level(logging.WARNING):
            stat = raster.get_stats(stats_name="80 percentile")
            assert isnan(stat)
        assert "Statistic name '80 percentile' is not recognized" in caplog.text

        with caplog.at_level(logging.WARNING):
            stat = raster.get_stats(stats_name=42)
            assert stat is None
        # assert "Statistic name '42' is not recognized string" in caplog.text

        # Empty mask (=False)
        inlier_mask = ~raster.get_mask()
        empty_mask = np.zeros_like(inlier_mask)
        raster.set_mask(~empty_mask)
        with caplog.at_level(logging.WARNING):
            stats_masked = raster.to_pointcloud().get_stats()
        assert "Empty raster, returns Nan for all stats" in caplog.text
        assert len(stats_masked) == len(expected_stats + expected_stats_count)
        for name in expected_stats:
            assert np.isnan(stats_masked.get(name))
        assert stats_masked.get("Valid count") == 0
        assert stats_masked.get("Total count") == 0
        assert isnan(stats_masked.get("Percentage valid points"))

    def test_raster_get_stats_values(self) -> None:
        """
        Verify the output statistics values of a raster.
        """
        filename_rast = gu.examples.get_path("everest_landsat_b4")
        filename_vect = gu.examples.get_path("everest_rgi_outlines")
        rast = gu.Raster(filename_rast)
        vect = gu.Vector(filename_vect)
        inlier_mask = ~vect.create_mask(rast)

        # Verify raster stats
        res_stats = {
            "Mean": np.float64(144.04460496183205),
            "Median": np.float64(124.0),
            "Max": np.uint8(255),
            "Min": np.uint8(13),
            "Sum": np.uint64(75479373),
            "Sum of squares": np.uint64(44549637),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(164.0),
            "NMAD": np.float64(94.8864),
            "RMSE": np.float64(9.220541807365446),
            "Standard deviation": np.float64(79.44349437534403),
            "Valid count": 524000,
            "Total count": 524000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(res_stats, rast.get_stats())

        # Verify raster stats with a mask
        res_stats_mask = {
            "Mean": np.float64(110.49218069801574),
            "Median": np.float64(92.0),
            "Max": np.uint8(255),
            "Min": np.uint8(13),
            "Sum": np.uint64(26650493),
            "Sum of squares": np.uint64(24696991),
            "90th percentile": np.float64(225.0),
            "LE90": np.float64(223.0),
            "IQR": np.float64(83.0),
            "NMAD": np.float64(54.856199999999994),
            "RMSE": np.float64(10.118943490060417),
            "Standard deviation": np.float64(64.98157041836747),
            "Valid count": 524000,
            "Total count": 524000,
            "Percentage valid points": np.float64(100.0),
            "Valid inlier count": np.int64(241198),
            "Total inlier count": np.int64(241198),
            "Percentage inlier points": np.float64(46.03015267175572),
            "Percentage valid inlier points": np.float64(100.0),
        }
        compare_dict(res_stats_mask, rast.get_stats(inlier_mask=inlier_mask))

        # Verify cropped raster
        nrows, ncols = rast.shape
        rast_crop = rast.icrop((100, 100, ncols - 100, nrows - 100))
        res_stats_crop = {
            "Mean": np.float64(148.69901465201465),
            "Median": np.float64(133.0),
            "Max": np.uint8(255),
            "Min": np.uint8(14),
            "Sum": np.uint64(40594831),
            "Sum of squares": np.uint64(22875807),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(166.0),
            "NMAD": np.float64(105.26459999999999),
            "RMSE": np.float64(9.153915273540871),
            "Standard deviation": np.float64(79.32951386752386),
            "Valid count": 273000,
            "Total count": 273000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(res_stats_crop, rast_crop.get_stats())

        # Verify reprojected raster
        rast_crop_proj = rast_crop.reproject(rast, nodata=255, resampling=rio.warp.Resampling.nearest)
        res_stats_crop_proj = {
            "Mean": np.float64(117.80631314205752),
            "Median": np.float64(107.0),
            "Max": np.uint8(254),
            "Min": np.uint8(14),
            "Sum": np.uint64(24919216),
            "Sum of squares": np.uint64(22814334),
            "90th percentile": np.float64(218.0),
            "LE90": np.float64(204.0),
            "IQR": np.float64(93.0),
            "NMAD": np.float64(66.717),
            "RMSE": np.float64(10.38534653788665),
            "Standard deviation": np.float64(62.319986152883956),
            "Valid count": 211527,
            "Total count": 524000,
            "Percentage valid points": np.float64(40.36774809160305),
        }
        print(rast_crop_proj.get_stats())
        compare_dict(res_stats_crop_proj, rast_crop_proj.get_stats())

        # Verify stats of a masked raster
        rast.set_mask(inlier_mask)
        stats_masked_rast = {
            "Mean": np.float64(172.66101371277432),
            "Median": np.float64(188.0),
            "Max": np.uint8(255),
            "Min": np.uint8(15),
            "Sum": np.uint64(48828880),
            "Sum of squares": np.uint64(19852646),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(209.0),
            "IQR": np.float64(156.0),
            "NMAD": np.float64(99.3342),
            "RMSE": np.float64(8.37853254688833),
            "Standard deviation": np.float64(79.45825061580675),
            "Valid count": 282802,
            "Total count": 524000,
            "Percentage valid points": np.float64(53.96984732824428),
        }
        compare_dict(stats_masked_rast, rast.get_stats())

        # Verify stats of a masked raster with the other part covered by the inler_mask (=> empty raster)
        stats_masked_rast_masked = {
            "Mean": np.nan,
            "Median": np.nan,
            "Max": np.nan,
            "Min": np.nan,
            "Sum": np.nan,
            "Sum of squares": np.nan,
            "90th percentile": np.nan,
            "LE90": np.nan,
            "IQR": np.nan,
            "NMAD": np.nan,
            "RMSE": np.nan,
            "Standard deviation": np.nan,
            "Valid count": 282802,
            "Total count": 524000,
            "Percentage valid points": np.float64(53.96984732824428),
            "Valid inlier count": np.int64(0),
            "Total inlier count": np.int64(241198),
            "Percentage inlier points": np.float64(0.0),
            "Percentage valid inlier points": np.float64(0.0),
        }
        compare_dict(stats_masked_rast_masked, rast.get_stats(inlier_mask=inlier_mask))

    def test_pointcloud_get_stats_values(self) -> None:
        """
        Verify the output statistics values of a pointcloud.
        """

        filename_rast = gu.examples.get_path("everest_landsat_b4")
        rast = gu.Raster(filename_rast)
        rast_pc = rast.to_pointcloud()

        # Verify pc stats
        rast_stats_pc = {
            "Mean": np.float64(144.04460496183205),
            "Median": np.float64(124.0),
            "Max": np.uint8(255),
            "Min": np.uint8(13),
            "Sum": np.uint64(75479373),
            "Sum of squares": np.uint64(44549637),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(164.0),
            "NMAD": np.float64(94.8864),
            "RMSE": np.float64(9.220541807365446),
            "Standard deviation": np.float64(79.44349437534403),
            "Valid count": 524000,
            "Total count": 524000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(rast_stats_pc, rast_pc.get_stats())

        # Verify cropped raster pc
        nrows, ncols = rast.shape
        rast_crop = rast.icrop((100, 100, ncols - 100, nrows - 100))
        rast_crop_pc = rast_crop.to_pointcloud()

        rast_stats_crop_pc = {
            "Mean": np.float64(148.69901465201465),
            "Median": np.float64(133.0),
            "Max": np.uint8(255),
            "Min": np.uint8(14),
            "Sum": np.uint64(40594831),
            "Sum of squares": np.uint64(22875807),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(166.0),
            "NMAD": np.float64(105.26459999999999),
            "RMSE": np.float64(9.153915273540871),
            "Standard deviation": np.float64(79.32951386752386),
            "Valid count": 273000,
            "Total count": 273000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(rast_stats_crop_pc, rast_crop_pc.get_stats())

        # Verify reprojected raster pc
        rast_crop_proj = rast_crop.reproject(rast, nodata=255, resampling=rio.warp.Resampling.nearest)
        for data in [rast_crop_proj, rast_crop, rast]:
            print(
                "# data shape:", data.shape, data.shape[0] * rast_crop.shape[1], "->", len(data.to_pointcloud()["b1"])
            )
            print("    ", data.to_pointcloud()["b1"].min(), "to", data.to_pointcloud()["b1"].max())
        rast_crop_proj_pc = rast_crop_proj.to_pointcloud()

        rast_stats_crop_proj_pc = {
            "Mean": np.float64(117.80631314205752),
            "Median": np.float64(107.0),
            "Max": np.uint8(254),
            "Min": np.uint8(14),
            "Sum": np.uint64(24919216),
            "Sum of squares": np.uint64(22814334),
            "90th percentile": np.float64(218.0),
            "LE90": np.float64(204.0),
            "IQR": np.float64(93.0),
            "NMAD": np.float64(66.717),
            "RMSE": np.float64(10.38534653788665),
            "Standard deviation": np.float64(62.319986152883956),
            "Valid count": 211527,
            "Total count": 211527,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(rast_stats_crop_proj_pc, rast_crop_proj_pc.get_stats())
