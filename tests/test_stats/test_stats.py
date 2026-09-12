from __future__ import annotations

import warnings
from cmath import isnan
from typing import Any

import numpy as np
import pandas as pd
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.stats.reduction import (
    _STATS_ALIAS_ALL,
    _STATS_ALIAS_CALLABLE,
    _STATS_ALIAS_GEN,
    _STATS_ALIAS_MASK,
    _STATS_LIST_MIN,
)

stat_types = (int, float, np.integer, np.floating)


def compare_dict(dict1: dict, dict2: dict) -> None:  # type: ignore
    assert dict1.keys() == dict2.keys()
    for key in dict1.keys():
        assert key in dict2
        if dict1[key] is not np.nan:
            assert dict2[key] == pytest.approx(dict1[key], abs=1e-10)
        else:
            assert isnan(dict2[key])


class TestStats:
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_stats__summary_and_grouped_routes(self) -> None:
        """Checks that by chooses between a summary and a grouped table."""

        # Create a small raster and two declared land-cover categories
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        landcover = np.array([[0, 0], [1, 1]])
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 2, 1, 1),
            crs=4326,
        )

        # Calculate the default, scalar, and selected summaries through the common method
        summary = raster.stats()
        mean = raster.stats("mean")
        selected = raster.stats(["mean", "std", "nmad"])
        array_mean = gu.stats.stats(values, "mean")

        # Calculate the same selected statistics separately for each category
        grouped = raster.stats(
            ["mean", "std"],
            by={"landcover": landcover},
            categories={"landcover": [0, 1]},
        )
        array_grouped = gu.stats.stats(
            values,
            ["mean", "std"],
            by={"landcover": landcover},
            categories={"landcover": [0, 1]},
        )
        default_grouped = raster.stats(
            by={"landcover": landcover},
            categories={"landcover": [0, 1]},
        )

        # Check the summary, grouped routes, and shared default statistic selection
        assert summary["Mean"] == pytest.approx(values.mean())
        assert mean == pytest.approx(values.mean())
        assert array_mean == pytest.approx(values.mean())
        assert selected == pytest.approx({"mean": values.mean(), "std": values.std(), "nmad": 1.4826})
        np.testing.assert_allclose(grouped["band_1"], [[2, 1.5, 0.5], [2, 3.5, 0.5]])
        np.testing.assert_allclose(array_grouped["value"], grouped["band_1"])
        expected_grouped_statistics = ["count", *[name for name in _STATS_LIST_MIN if name != "validcount"]]
        assert default_grouped.columns.get_level_values("statistic").tolist() == expected_grouped_statistics

    @pytest.mark.parametrize("subsampling_strategy", ["topk", "sequential"])
    def test_stats__subsample_per_group_without_by(self, subsampling_strategy: str) -> None:
        """Checks that per-group sampling without groups behaves like ordinary summary sampling."""

        # Use a shared mask so the sample must exclude locations before selecting any values
        values = pd.Series(np.arange(30, dtype=float))
        mask = values.to_numpy() % 3 != 0
        options = {"mask": mask, "subsample": 7, "random_state": 42, "subsampling_strategy": subsampling_strategy}

        # Compare the option against the established summary path with the same seed
        expected = gu.stats.stats(values, ["mean", "std", "totalcount"], **options)
        result = gu.stats.stats(values, ["mean", "std", "totalcount"], subsample_per_group=True, **options)
        assert result == expected
        assert result["totalcount"] == 7

    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize("invalid", ["False", 1, None])
    def test_stats__error_subsample_per_group_validation(self, grouped: bool, invalid: object) -> None:
        """Checks that both stats() routes reject non-boolean per-group sampling options."""

        # A string such as 'False' must not accidentally enable sampling within groups
        values = pd.Series(np.arange(4, dtype=float))
        by = {"group": values > 1} if grouped else None
        with pytest.raises(TypeError, match="Argument ``subsample_per_group`` must be a boolean"):
            gu.stats.stats(values, "mean", by=by, subsample_per_group=invalid)

    @pytest.mark.parametrize("as_list", [False, True])
    def test_stats__empty_mask_callable_result(self, as_list: bool) -> None:
        """Checks that a callable's result stays masked when no selected values remain."""

        # NumPy Masked mean returns a masked scalar for an entirely excluded array
        values = np.arange(6, dtype=float).reshape(2, 3)
        keep = np.zeros(values.shape, dtype=bool)
        statistics = [np.ma.mean] if as_list else np.ma.mean

        # Check the scalar and named-result routes without allowing scalar conversion to expose masked storage
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Empty raster")
            result = gu.stats.stats(values, statistics, mask=keep)
        assert np.ma.is_masked(result["mean"] if as_list else result)

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__generator_for_multiple_values(self, grouped: bool) -> None:
        """Checks that a generator requests the same statistics for every selected value array."""

        # Give the second named array a distinct mean and the exact same spread
        first = np.arange(6, dtype=float)
        values = {"first": first, "second": first + 100}
        options = {}
        if grouped:
            options = {"by": {"zone": np.array([0, 0, 0, 1, 1, 1])}, "categories": {"zone": [0, 1]}}

        # Pass a generator that can only be consumed once while selecting both arrays
        statistics = (name for name in ["mean", "std"])
        result = gu.stats.stats(values, statistics, **options)

        # Check each array independently so an exhausted request cannot silently omit its statistics
        for name, array in values.items():
            if grouped:
                expected = [[3, array[:3].mean(), array[:3].std()], [3, array[3:].mean(), array[3:].std()]]
                np.testing.assert_allclose(result[name], expected)
            else:
                assert result[name] == pytest.approx({"mean": array.mean(), "std": array.std()})

    @pytest.mark.parametrize("source_type", ["raster", "pointcloud"])
    def test_get_stats__deprecated_forwarding(self, source_type: str) -> None:
        """Checks that get_stats() warns about deprecation and forwards its selections to stats()."""

        # Select a second raster band and a mask, or the active values of a point cloud
        values = np.arange(1, 7, dtype=float).reshape(2, 3)
        statistics = ["mean", "std"]
        old_options: dict[str, Any] = {"stats_name": statistics}
        new_options: dict[str, Any] = {"statistics": statistics}
        if source_type == "raster":
            source = gu.Raster.from_array(np.stack((values, values + 100)), rio.transform.from_origin(0, 2, 1, 1), 4326)
            keep = np.array([[True, True, False], [True, False, True]])
            old_options.update(band=2, inlier_mask=keep)
            new_options.update(values=2, mask=keep)
        else:
            source = gu.PointCloud.from_xyz(np.arange(values.size), np.zeros(values.size), values.ravel(), crs=4326)

        # Require an explicit deprecation warning while comparing with the canonical API
        expected = source.stats(**new_options)
        with pytest.deprecated_call(match="get_stats"):
            deprecated = source.get_stats(**old_options)

        # Check that the requested statistic names and all numerical results are unchanged
        compare_dict(expected, deprecated)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    def test_stats__raster_one_band(self, example: str) -> None:
        """Checks raster stats for various statistic names, masks, callables and empty data."""
        raster = gu.Raster(example)

        # Default stats
        stats = raster.stats()
        assert len(stats) == len(_STATS_LIST_MIN)
        assert list(stats.keys()) == [_STATS_ALIAS_ALL[key] for key in _STATS_LIST_MIN]
        for name in _STATS_LIST_MIN:
            assert _STATS_ALIAS_ALL[name] in stats
            assert isinstance(stats.get(_STATS_ALIAS_ALL[name]), stat_types)

        # Full stats
        stats = raster.stats("all")
        assert len(stats) == len(_STATS_ALIAS_GEN)
        for name in _STATS_ALIAS_GEN.values():
            assert name in stats
            assert isinstance(stats.get(name), stat_types)

        # With mask (inlier=True)
        inlier_mask = ~raster.get_mask()
        stats_masked = raster.stats("all", mask=inlier_mask)
        assert len(stats_masked) == len(_STATS_ALIAS_ALL)
        assert list(stats_masked.keys()) == [_STATS_ALIAS_ALL[key] for key in _STATS_ALIAS_ALL]
        for name in _STATS_ALIAS_MASK.values():
            assert name in stats_masked
            stats_masked.pop(name)
        assert stats_masked == stats

        # Print of the values
        stats = raster.stats("all", mask=inlier_mask)
        for stat in stats:
            assert not isinstance(stat, np.generic)
        for stat in _STATS_ALIAS_ALL:
            assert not isinstance(raster.stats(stat, mask=inlier_mask), np.generic)

        # With mask (inlier=True) and default list
        stats_masked = raster.stats(mask=inlier_mask)
        assert len(stats_masked) == len(_STATS_LIST_MIN)
        assert list(stats_masked.keys()) == [_STATS_ALIAS_ALL[key] for key in _STATS_LIST_MIN]
        for name in _STATS_LIST_MIN:
            assert _STATS_ALIAS_ALL[name] in stats_masked

        # Test case sensitive + space/underscore possibilities
        stats_masked = raster.stats(mask=inlier_mask)
        name = "Standard deviation"
        assert stats_masked["Standard deviation"] == raster.stats(statistics="standard deviation", mask=inlier_mask)
        assert stats_masked["Standard deviation"] == raster.stats(statistics="standarddeviation", mask=inlier_mask)
        assert stats_masked["Standard deviation"] == raster.stats(statistics="standard_deviation", mask=inlier_mask)
        assert stats_masked[name] == raster.stats(statistics="standard_deviation", mask=inlier_mask)

        # Empty mask (=False)
        empty_mask = np.zeros_like(inlier_mask)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            stats_masked = raster.stats("all", mask=empty_mask)
        assert len(stats_masked) == len(_STATS_ALIAS_ALL)
        for name in _STATS_ALIAS_CALLABLE.values():
            assert np.isnan(stats_masked.get(name))

        assert stats_masked.get("Valid count") == stats.get("Valid count")
        assert stats_masked.get("Total count") == stats.get("Total count")
        assert stats_masked.get("Percentage valid points") == stats.get("Percentage valid points")

        for stat in _STATS_ALIAS_ALL:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
                stats_masked = raster.stats(mask=empty_mask, statistics=stat)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            stats_masked = raster.stats(mask=empty_mask, statistics="mean")
        assert np.isnan(stats_masked)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            stats_masked = raster.stats(mask=empty_mask, statistics="valid_count")
        assert stats_masked == stats.get("Valid count")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            stats_masked = raster.stats(mask=empty_mask, statistics="Valid inlier count")
        assert stats_masked == 0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            stats_masked = raster.stats("all", mask=inlier_mask)
            for name in stats_masked:
                assert stats_masked[name] == raster.stats(statistics=name.lower(), mask=inlier_mask)
                assert stats_masked[name] == raster.stats(statistics="".join(name.split()), mask=inlier_mask)

        # Empty DEM
        dem_empty = gu.Raster.from_array(
            np.random.randint(42, size=(0, 0), dtype="uint8"),
            transform=rio.transform.from_origin(10, 20, 1, 1),
            crs=4326,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            stats_empty = dem_empty.stats("all")
        assert len(stats_empty) == len(_STATS_ALIAS_GEN)
        for name in _STATS_ALIAS_CALLABLE.values():
            assert np.isnan(stats_empty.get(name))
        assert stats_empty.get("Valid count") == 0
        assert stats_empty.get("Total count") == 0
        assert isnan(stats_empty.get("Percentage valid points"))

        # Single stat
        for name in _STATS_ALIAS_GEN:
            stat = raster.stats(statistics=name)
            assert np.isfinite(stat)
        for name in _STATS_ALIAS_MASK:
            stat = raster.stats(statistics=name)
            assert np.isnan(stat)

        # Alias stat
        assert raster.stats(statistics="Valid count") == raster.stats(statistics="valid_count")

        # Callable
        def percentile_95(data: NDArrayNum) -> np.floating[Any]:
            if isinstance(data, np.ma.MaskedArray):
                data = data.compressed()
            return np.nanpercentile(data, 95)

        stat = raster.stats(statistics=percentile_95)
        assert isinstance(stat, np.floating)

        # Selected stats and callable
        stats_name = ["mean", "max", "std", "validinliercount", "percentile_95"]
        stats = raster.stats(statistics=["mean", "max", "std", "validinliercount", percentile_95], mask=inlier_mask)
        assert len(stats) == len(stats_name)
        for name in stats_name:
            assert name in stats
            assert not isnan(stats.get(name))

        # Non-existing stats
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Statistic name 80 percentile is not recognized")
            stat = raster.stats(statistics="80 percentile")
        assert isnan(stat)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Statistic name 42 is a not recognized string")
            stat = raster.stats(statistics=42)
        assert stat is None

        # IQR (scipy) validation with numpy
        nan_arr = raster.get_nanarray()
        if nan_arr.ndim == 3:
            nan_arr = nan_arr[0, :, :]
        assert raster.stats(statistics="iqr") == pytest.approx(
            np.nanpercentile(nan_arr, 75) - np.nanpercentile(nan_arr, 25)
        )

    @pytest.mark.parametrize("example", [landsat_rgb_path])
    def test_stats__multiple_raster_bands(self, example: str) -> None:
        """Checks that stats() calculates each band of a multiband raster separately."""

        raster = gu.Raster(example)
        stats = raster.stats()
        assert list(stats.keys()) == ["band 1", "band 2", "band 3"]
        data = raster.get_nanarray()
        for band in range(1, raster.count + 1):
            assert stats["band " + str(band)]["Mean"] == pytest.approx(np.nanmean(data[band - 1]))

        stats = raster.stats("mean")
        for band in range(1, raster.count + 1):
            assert stats["band " + str(band)] == pytest.approx(np.nanmean(data[band - 1]))

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])
    def test_stats__raster_pointcloud(self, example: str) -> None:
        """Checks statistics for a raster converted to a point cloud."""
        raster = gu.Raster(example)
        pointcloud = raster.to_pointcloud()

        # Default stats
        stats = pointcloud.stats()
        assert len(stats) == len(_STATS_LIST_MIN)
        assert list(stats.keys()) == [_STATS_ALIAS_ALL[key] for key in _STATS_LIST_MIN]
        for name in _STATS_LIST_MIN:
            assert _STATS_ALIAS_ALL[name] in stats
            assert isinstance(stats.get(_STATS_ALIAS_GEN[name]), stat_types)

        # Full stats
        stats = pointcloud.stats("all")
        assert len(stats) == len(_STATS_ALIAS_GEN)
        assert list(stats.keys()) == [_STATS_ALIAS_GEN[key] for key in _STATS_ALIAS_GEN]
        for name in _STATS_ALIAS_GEN.values():
            assert name in stats
            assert isinstance(stats.get(name), stat_types)

        # Single stat
        for name in _STATS_ALIAS_GEN:
            stat = pointcloud.stats(statistics=name)
            assert np.isfinite(stat)
        for name in _STATS_ALIAS_MASK:
            stat = pointcloud.stats(statistics=name)
            assert np.isnan(stat)

        # Print of the values
        stats = pointcloud.stats("all")
        for stat in stats:
            assert not isinstance(stat, np.generic)
        for stat in _STATS_ALIAS_ALL:
            assert not isinstance(pointcloud.stats(stat), np.generic)

        # Callable
        def percentile_95(data: NDArrayNum) -> np.floating[Any]:
            if isinstance(data, np.ma.MaskedArray):
                data = data.compressed()
            return np.nanpercentile(data, 95)

        # Selected stats and callable
        stats_name = ["mean", "max", "std", "percentile_95"]
        stats = pointcloud.stats(statistics=["mean", "max", "std", percentile_95])
        assert len(stats) == len(stats_name)
        for name in stats_name:
            assert name in stats
            assert stats.get(name) is not None

        # Non-existing stats
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Statistic name 80 percentile is not recognized")
            stat = pointcloud.stats(statistics="80 percentile")
        assert isnan(stat)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Statistic name 42 is a not recognized string")
            stat = pointcloud.stats(statistics=42)
        assert stat is None

        # Empty mask (=False)
        inlier_mask = ~raster.get_mask()
        inlier_mask = ~raster.get_mask()
        empty_mask = np.zeros_like(inlier_mask)
        raster.set_mask(~empty_mask)
        pointcloud = raster.to_pointcloud()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Empty raster")
            stats_masked = pointcloud.stats("all")
        assert len(stats_masked) == len(_STATS_ALIAS_GEN)
        for name in _STATS_ALIAS_CALLABLE.values():

            assert np.isnan(stats_masked.get(name))
        assert stats_masked.get("Valid count") == 0
        assert stats_masked.get("Total count") == 0
        assert isnan(stats_masked.get("Percentage valid points"))

    def test_stats__raster_values(self) -> None:
        """Checks the output statistics values of a raster."""
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
            "Sum of squares": np.uint64(14179501317),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(164.0),
            "NMAD": np.float64(94.8864),
            "RMSE": np.float64(164.49959579638966),
            "Standard deviation": np.float64(79.44349437534403),
            "Valid count": 524000,
            "Total count": 524000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(res_stats, rast.stats("all"))

        # Verify raster stats with a mask
        res_stats_mask = {
            "Mean": np.float64(110.49218069801574),
            "Median": np.float64(92.0),
            "Max": np.uint8(255),
            "Min": np.uint8(13),
            "Sum": np.uint64(26650493),
            "Sum of squares": np.uint64(3963154847),
            "90th percentile": np.float64(225.0),
            "LE90": np.float64(223.0),
            "IQR": np.float64(83.0),
            "NMAD": np.float64(54.856199999999994),
            "RMSE": np.float64(128.18395566310244),
            "Standard deviation": np.float64(64.98157041836747),
            "Valid count": 524000,
            "Total count": 524000,
            "Percentage valid points": np.float64(100.0),
            "Valid inlier count": np.int64(241198),
            "Total inlier count": np.int64(241198),
            "Percentage inlier points": np.float64(46.03015267175572),
            "Percentage valid inlier points": np.float64(100.0),
        }
        compare_dict(res_stats_mask, rast.stats("all", mask=inlier_mask))

        # Verify cropped raster
        nrows, ncols = rast.shape
        rast_crop = rast.icrop((100, 100, ncols - 100, nrows - 100))
        res_stats_crop = {
            "Mean": np.float64(148.69901465201465),
            "Median": np.float64(133.0),
            "Max": np.uint8(255),
            "Min": np.uint8(14),
            "Sum": np.uint64(40594831),
            "Sum of squares": np.uint64(7754447263),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(166.0),
            "NMAD": np.float64(105.26459999999999),
            "RMSE": np.float64(168.53655012767328),
            "Standard deviation": np.float64(79.32951386752386),
            "Valid count": 273000,
            "Total count": 273000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(res_stats_crop, rast_crop.stats("all"))

        # Verify reprojected raster
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="New nodata.*")
            rast_crop.set_nodata(255)  # Needs to be defined for reprojection
        rast_crop_proj = rast_crop.reproject(rast, resampling=rio.warp.Resampling.nearest)
        res_stats_crop_proj = {
            "Mean": np.float64(117.80631314205752),
            "Median": np.float64(107.0),
            "Max": np.uint8(254),
            "Min": np.uint8(14),
            "Sum": np.uint64(24919216),
            "Sum of squares": np.uint64(3757165438),
            "90th percentile": np.float64(218.0),
            "LE90": np.float64(204.0),
            "IQR": np.float64(93.0),
            "NMAD": np.float64(66.717),
            "RMSE": np.float64(133.27455905093126),
            "Standard deviation": np.float64(62.319986152883956),
            "Valid count": 211527,
            "Total count": 524000,
            "Percentage valid points": np.float64(40.36774809160305),
        }
        compare_dict(res_stats_crop_proj, rast_crop_proj.stats("all"))

        # Verify stats of a masked raster
        rast.set_mask(inlier_mask)
        stats_masked_rast = {
            "Mean": np.float64(172.66101371277432),
            "Median": np.float64(188.0),
            "Max": np.uint8(255),
            "Min": np.uint8(15),
            "Sum": np.uint64(48828880),
            "Sum of squares": np.uint64(10216346470),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(209.0),
            "IQR": np.float64(156.0),
            "NMAD": np.float64(99.3342),
            "RMSE": np.float64(190.06693359773863),
            "Standard deviation": np.float64(79.45825061580675),
            "Valid count": 282802,
            "Total count": 524000,
            "Percentage valid points": np.float64(53.96984732824428),
        }
        compare_dict(stats_masked_rast, rast.stats("all"))

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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="Empty raster")
            compare_dict(stats_masked_rast_masked, rast.stats("all", mask=inlier_mask))

    def test_stats__pointcloud_values(self) -> None:
        """Checks the output statistics values of a pointcloud."""

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
            "Sum of squares": np.uint64(14179501317),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(164.0),
            "NMAD": np.float64(94.8864),
            "RMSE": np.float64(164.49959579638966),
            "Standard deviation": np.float64(79.44349437534403),
            "Valid count": 524000,
            "Total count": 524000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(rast_stats_pc, rast_pc.stats("all"))

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
            "Sum of squares": np.uint64(7754447263),
            "90th percentile": np.float64(255.0),
            "LE90": np.float64(218.0),
            "IQR": np.float64(166.0),
            "NMAD": np.float64(105.26459999999999),
            "RMSE": np.float64(168.53655012767328),
            "Standard deviation": np.float64(79.32951386752386),
            "Valid count": 273000,
            "Total count": 273000,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(rast_stats_crop_pc, rast_crop_pc.stats("all"))

        # Verify reprojected raster pc
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="New nodata.*")
            rast_crop.set_nodata(255)  # Needs to be defined for reprojection
        rast_crop_proj = rast_crop.reproject(rast, resampling=rio.warp.Resampling.nearest)
        rast_crop_proj_pc = rast_crop_proj.to_pointcloud()

        rast_stats_crop_proj_pc = {
            "Mean": np.float64(117.80631314205752),
            "Median": np.float64(107.0),
            "Max": np.uint8(254),
            "Min": np.uint8(14),
            "Sum": np.uint64(24919216),
            "Sum of squares": np.uint64(3757165438),
            "90th percentile": np.float64(218.0),
            "LE90": np.float64(204.0),
            "IQR": np.float64(93.0),
            "NMAD": np.float64(66.717),
            "RMSE": np.float64(133.27455905093126),
            "Standard deviation": np.float64(62.319986152883956),
            "Valid count": 211527,
            "Total count": 211527,
            "Percentage valid points": np.float64(100.0),
        }
        compare_dict(rast_stats_crop_proj_pc, rast_crop_proj_pc.stats("all"))


class TestStatsChunked:
    """Checks stats() loading behavior and exact results with chunked inputs."""

    @pytest.mark.parametrize("masked", [False, True])
    def test_stats__multiprocessing_summary(self, masked: bool) -> None:
        """Checks that stats() combines ungrouped array tiles and optional masks in worker processes."""

        # Include nodata values and uneven edge tiles in both the array and raster routes
        from geoutils.multiproc.cluster import MpCluster

        values = np.arange(35, dtype=float).reshape(5, 7)
        values[1, 2] = np.nan
        values[3, 5] = np.nan
        raster = gu.Raster.from_array(
            values,
            transform=rio.transform.from_origin(0, 5, 1, 1),
            crs=4326,
            nodata=np.nan,
        )
        keep = np.indices(values.shape).sum(axis=0) % 3 != 0 if masked else None
        expected_summary = gu.stats.stats(values, mask=keep)
        expected_all = gu.stats.stats(values, "all", mask=keep)
        expected_reductions = gu.stats.stats(values, ["mean", "std", "sum", "validcount"], mask=keep)

        # Run exact default statistics and mergeable reductions through real worker processes
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=(2, 3), cluster=cluster)
            array_summary = gu.stats.stats(values, mask=keep, mp_config=config)
            array_all = gu.stats.stats(values, "all", mask=keep, mp_config=config)
            raster_summary = raster.stats(mask=keep, mp_config=config)
            reductions = raster.stats(["mean", "std", "sum", "validcount"], mask=keep, mp_config=config)

        # Match the eager values and preserve integer count results in both public APIs
        compare_dict(expected_summary, array_summary)
        compare_dict(expected_all, array_all)
        compare_dict(expected_summary, raster_summary)
        compare_dict(expected_reductions, reductions)
        assert isinstance(array_summary["Valid count"], int)
        assert isinstance(array_summary["Total count"], int)

    def test_stats__custom_summary_preserves_shape(self) -> None:
        """Checks that Multiproc custom summaries match eager results and see the complete array shape."""

        # Use a rectangular array whose full shape differs from every worker tile and from a flattened array
        values = np.arange(12, dtype=float).reshape(3, 4)
        config = MultiprocConfig(chunks=(2, 3))

        # Request the same custom summary eagerly and through Multiproc chunks
        expected_shape = gu.stats.stats(values, np.shape)
        expected_combined = gu.stats.stats(values, [np.shape, "mean"])
        shape = gu.stats.stats(values, np.shape, mp_config=config)
        combined = gu.stats.stats(values, [np.shape, "mean"], mp_config=config)

        # Preserve the complete shape and exactly match both eager output forms
        assert shape == expected_shape
        assert combined == expected_combined
        assert shape == combined["shape"] == values.shape
        assert combined["mean"] == pytest.approx(values.mean())
