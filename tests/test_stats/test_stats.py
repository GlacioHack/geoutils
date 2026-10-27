from __future__ import annotations

import warnings
from cmath import isnan
from contextlib import nullcontext
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples
from geoutils._dispatch import is_dask_array
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.operators.reducer import Mean, Mode, Quantile, Range, Sum
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
            assert dict2[key] == pytest.approx(dict1[key])
        else:
            assert isnan(dict2[key])


class TestStats:
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_stats__reducer_objects(self) -> None:
        """Checks that stats() reuses Reducers globally and within declared groups."""

        # Split four finite values into two categories with a known mean and range in each group
        values = np.array([[1.0, 3.0], [6.0, 10.0]])
        groups = np.array([[0, 0], [1, 1]])

        # A built-in linear Reducer follows the established optimized statistic result
        mean = gu.stats.stats(values, Mean())
        assert mean == gu.stats.stats(values, "mean")
        total = gu.stats.stats(values, Sum())
        assert total == gu.stats.stats(values, "sum")

        # A nonlinear Reducer without a legacy statistic name receives each complete group through LocalData
        grouped = gu.stats.stats(
            values,
            [
                Mean(),
                Range(),
                Mode(weighted=False, tie_break="first"),
                Quantile(0.25),
            ],
            by={"group": groups},
            categories={"group": [0, 1]},
        )
        np.testing.assert_array_equal(grouped["value", "Mean"], [2.0, 8.0])
        np.testing.assert_array_equal(grouped["value", "Range"], [2.0, 4.0])
        np.testing.assert_array_equal(grouped["value", "Mode"], [1.0, 6.0])
        np.testing.assert_array_equal(grouped["value", "Quantile"], [1.5, 7.0])

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
    def test_stats__subsample_per_group_without_by(self, subsampling_strategy: Literal["topk", "sequential"]) -> None:
        """Checks that per-group sampling without groups behaves like ordinary summary sampling."""

        # Use a shared mask so the sample must exclude locations before selecting any values
        values = pd.Series(np.arange(30, dtype=float))
        mask = values.to_numpy() % 3 != 0

        # Compare the option against the established summary path with the same seed
        expected = gu.stats.stats(
            values,
            ["mean", "std", "totalcount"],
            mask=mask,
            subsample=7,
            random_state=42,
            subsampling_strategy=subsampling_strategy,
        )
        result = gu.stats.stats(
            values,
            ["mean", "std", "totalcount"],
            mask=mask,
            subsample=7,
            random_state=42,
            subsampling_strategy=subsampling_strategy,
            subsample_per_group=True,
        )
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
            gu.stats.stats(values, "mean", by=by, subsample_per_group=invalid)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("option", "message", "error"),
        [
            ({"backend": "invalid"}, "Argument ``backend``", ValueError),
            ({"subsampling_strategy": "invalid"}, "Argument ``subsampling_strategy``", ValueError),
            ({"subsample": 0}, "Argument ``subsample``", ValueError),
            ({"fractional": "yes"}, "Argument ``fractional``", TypeError),
            ({"fractional": True}, "Fractional statistics require one vector grouper", ValueError),
            ({"return_masks": True}, "Argument ``by`` is required", ValueError),
        ],
    )
    def test_stats__error_invalid_options(self, option: dict[str, Any], message: str, error: type[Exception]) -> None:
        """Checks errors raised for various options."""

        values = np.arange(4, dtype=float)
        with pytest.raises(error, match=message):
            gu.stats.stats(values, "mean", **option)

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
        by = {"zone": np.array([0, 0, 0, 1, 1, 1])} if grouped else None
        categories = {"zone": [0, 1]} if grouped else None

        # Pass a generator that can only be consumed once while selecting both arrays
        statistics = (name for name in ["mean", "std"])
        result = gu.stats.stats(values, statistics, by=by, categories=categories)

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
        nan_arr = raster.to_nanarray()
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
        data = raster.to_nanarray()
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

    @pytest.mark.skipif(find_spec("crick") is not None, reason="Only runs if crick is missing.")
    @pytest.mark.parametrize("backend", ["geoutils", "flox"])
    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize("statistics", ["all", ["median", "nmad", "90thpercentile", "iqr", "le90"]])
    def test_stats__quantiles_crick(
        self,
        backend: Literal["geoutils", "flox"],
        grouped: bool,
        statistics: str | list[str],
    ) -> None:
        """Checks that "auto" calculates exact in-memory quantiles (without crick)."""

        # Create nonlinear values with NaNs so quantile interpolation affects the result
        if backend == "flox":
            pytest.importorskip("flox")
        values = np.arange(1002, dtype=float) ** 2
        values[::19] = np.nan
        groups = np.arange(values.size) % 2
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1]}} if grouped else {}

        # We ensure having crick unavailable won't affect the run
        result = gu.stats.stats(values, statistics, backend=backend, **options)

        # We compare every quantile and spread exactly against NumPy on the original values
        names = ["median", "nmad", "90thpercentile", "iqr", "le90"]
        for label in range(2) if grouped else [0]:
            members = values[groups == label] if grouped else values
            members = members[np.isfinite(members)]
            lower, q25, median, q75, q90, upper = np.percentile(members, [5, 25, 50, 75, 90, 95])
            expected = [median, 1.4826 * np.median(np.abs(members - median)), q90, q75 - q25, upper - lower]
            if grouped:
                actual = [result.loc[label, ("value", name)] for name in names]
                assert "approximation" not in result.attrs["grouped_stats"]
            else:
                actual = [result[_STATS_ALIAS_ALL[name] if statistics == "all" else name] for name in names]
            np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("backend", ["geoutils", "flox"])
    @pytest.mark.parametrize("masked", [False, True])
    def test_stats__global_infinity(self, backend: Literal["geoutils", "flox"], masked: bool) -> None:
        """Checks edge case of infinity in global statistics."""

        # Create finite and infinite values
        if backend == "flox":
            pytest.importorskip("flox")
        values = np.array([1.0, np.inf])
        source = np.ma.array(values, mask=False) if masked else values
        statistics = ["mean", "sum", "min", "max", "validcount"]
        expected = gu.stats.stats(source, statistics)

        # Calculate statistics
        result = gu.stats.stats(source, statistics, backend=backend)

        # Check estimates and counts
        assert result == expected
        assert result["mean"] == np.inf
        assert result["sum"] == np.inf
        assert result["min"] == 1
        assert result["max"] == np.inf
        assert result["validcount"] == (2 if masked else 1)

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__flox_reduction(self, grouped: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that in-memory Flox statistics run independently of GeoUtils reductions."""

        # Create value arrays with NaNs and large integers
        pytest.importorskip("flox")
        from geoutils.stats import reduction

        values = {
            "height": np.array([1.0, 3.0, 10.0, np.nan, 2.0, 8.0, 9.0]),
            "code": 2**60 + np.arange(7, dtype=np.int64),
        }
        options: dict[str, Any] = (
            {"by": {"zone": np.arange(7) % 2}, "categories": {"zone": [0, 1, 2]}} if grouped else {}
        )
        statistics = ["mean", "std", "median", "nmad", "iqr", "rmse", "min", "max"]
        expected = gu.stats.stats(values, statistics, method="groupwise", **options)

        # Fail if the native reducer is called through the Flox backend
        def fail_native(*args: Any, **kwargs: Any) -> None:
            pytest.fail("Flox must perform its own reductions and merges")

        for name in ("_reduce_block", "_merge_blocks", "_aggregate_complete_groups"):
            monkeypatch.setattr(reduction, name, fail_native)
        result = gu.stats.stats(values, statistics, backend="flox", method="blockwise", **options)

        # Check equality
        if grouped:
            pd.testing.assert_frame_equal(result, expected)
            assert result[("code", "min")].min() == 2**60
        else:
            for value_name in values:
                for name in statistics:
                    np.testing.assert_allclose(result[value_name][name], expected[value_name][name])
            assert result["code"]["min"] == 2**60
            assert result["code"]["max"] == 2**60 + 6


class TestStatsUncertainty:
    """Test module for global and grouped statistics uncertainty and unsupported options."""

    def test_stats__propagates_mean(self) -> None:
        """Checks that a global mean propagates errors properly."""

        # Create 4 values with fixed error magnitude
        values = np.array([[1.0, 3.0], [6.0, 10.0]])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Derive stats, check mean value is equal
        nominal, summary = gu.stats.stats(
            values,
            Mean(),
            error_structure=source_error,
        )
        assert nominal == np.mean(values)
        np.testing.assert_array_equal(summary.estimate, [np.mean(values)])

        # Check the uncertainty (variance) on the mean value is as expected
        weights = np.full(values.size, 1 / values.size)
        expected_variance = np.sum((2 * weights) ** 2)
        np.testing.assert_allclose(summary.variance, [expected_variance])

    def test_stats__propagates_independently(self) -> None:
        """Checks that two value arrays have separate error propagation."""

        # We create two array with different values
        first = np.array([1.0, 3.0])
        second = np.array([10.0, 14.0])
        values = {"first": first, "second": second}
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # We propagate the mean for both, and check mean values
        nominal, summary = gu.stats.stats(
            values,
            Mean(),
            error_structure=source_error,
        )
        assert nominal == {name: np.mean(array) for name, array in values.items()}
        np.testing.assert_array_equal(summary.estimate, [np.mean(first), np.mean(second)])

        # We check output uncertainty
        np.testing.assert_allclose(summary.variance, [2, 2])

    def test_stats__propagates_grouped_mean(self) -> None:
        """Checks that grouped stats() propagates uncertainty properly."""

        # We create 2 groups of 2 values each, with and STD of 2
        values = np.array([[1.0, 3.0], [6.0, 10.0]])
        groups = np.array([[0, 0], [1, 1]])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])
        by = {"group": groups}
        categories = {"group": [0, 1]}

        # We estimate the grouped mean with/without uncertainty propagation
        expected = gu.stats.stats(values, Mean(), by=by, categories=categories)
        nominal, summary = gu.stats.stats(
            values,
            Mean(),
            error_structure=source_error,
            by=by,
            categories=categories,
        )
        pd.testing.assert_frame_equal(nominal, expected)
        np.testing.assert_array_equal(summary.estimate, [2.0, 8.0])

        # Each group mean has variance 4 / 2 from its two independent values
        np.testing.assert_allclose(summary.variance, [2.0, 2.0])

    @pytest.mark.parametrize(
        ("statistics", "option", "message", "error"),
        [
            ("mean", {}, "requires one Reducer", TypeError),
            (Mean(), {"subsample": 2}, "does not yet support subsampling", ValueError),
            (Mean(), {"backend": "flox"}, "requires backend='geoutils'", ValueError),
        ],
    )
    def test_stats__error_uncertainty(
        self, statistics: str | Mean, option: dict[str, Any], message: str, error: type[Exception]
    ) -> None:
        """Checks that uncertainty works only with proper reducer/backend/sampling scheme."""

        # We create an array with fixed error
        values = np.array([1.0, 3.0])
        source_error = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # We raise error on options not working with uncertainty
        with pytest.raises(error, match=message):
            gu.stats.stats(values, statistics, error_structure=source_error, **option)


class TestStatsChunked:
    """Checks stats() loading behavior and exact results with chunked inputs."""

    @pytest.mark.parametrize("backend,execution", [("geoutils", "dask"), ("geoutils", "mp"), ("flox", "dask")])
    @pytest.mark.parametrize("reindex", [None, False, True])
    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__map_reduce_chunk_invariance(
        self, backend: Literal["geoutils", "flox"], execution: str, reindex: bool | None, grouped: bool
    ) -> None:
        """Checks that map-reduce matches in-memory statistics."""

        # Create values with a NaN and groups crossing chunks
        if backend == "flox":
            pytest.importorskip("flox")
        if execution == "dask":
            da = pytest.importorskip("dask.array")

        values = np.arange(35, dtype=float).reshape(5, 7)
        values[1, 2] = np.nan
        groups = np.indices(values.shape).sum(axis=0) % 3
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1, 2, 3]}} if grouped else {}
        statistics = ["mean", "sum", "min", "max"]
        expected = gu.stats.stats(values, statistics, **options)
        source: Any = da.from_array(values, chunks=(2, 3)) if execution == "dask" else values.copy()

        # Run map-reduce with the selected options
        config = MultiprocConfig(chunks=(2, 3)) if execution == "mp" else None
        assert is_dask_array(source) == (execution == "dask")
        result = gu.stats.stats(
            source, statistics, backend=backend, method="map-reduce", reindex=reindex, mp_config=config, **options
        )

        # Check equality
        if grouped:
            exact_columns = result.columns.get_level_values("statistic") != "mean"
            pd.testing.assert_frame_equal(
                result.loc[:, exact_columns], expected.loc[:, exact_columns], check_exact=True
            )
            np.testing.assert_allclose(result[("value", "mean")], expected[("value", "mean")])
            assert result.attrs["grouped_stats"]["method"] == "map-reduce"
            assert result.attrs["grouped_stats"]["reindex"] == reindex
        else:
            assert result == expected
        assert is_dask_array(source) == (execution == "dask")
        np.testing.assert_array_equal(source.compute() if execution == "dask" else source, values)

    @pytest.mark.parametrize("backend,execution", [("geoutils", "dask"), ("geoutils", "mp"), ("flox", "dask")])
    @pytest.mark.parametrize("split_every", [None, 2, 3])
    @pytest.mark.parametrize("with_quantiles", [False, True])
    def test_stats__split_every(
        self,
        backend: Literal["geoutils", "flox"],
        execution: str,
        split_every: int | None,
        with_quantiles: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that split_every limits summary merges."""

        # Create constant groups with NaNs across multiple chunks
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")

        from geoutils.stats import reduction

        if with_quantiles:
            pytest.importorskip("crick")
        if backend == "flox":
            flox = pytest.importorskip("flox")
        groups = np.arange(141) % 2
        values = np.where(groups, 7.0, 2.0)
        values[:3] = np.nan
        statistics = ["mean", "sum", "min", "max"]
        if with_quantiles:
            statistics += ["median", "nmad"]
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1, 2]}}
        expected = gu.stats.stats(values, statistics, **options)
        sizes = []

        # Record merge sizes or the Flox configuration
        if backend == "geoutils":
            original_merge = reduction._merge_blocks

            def record_merge(summaries: Any) -> Any:
                sizes.append(len(summaries))
                return original_merge(summaries)

            monkeypatch.setattr(reduction, "_merge_blocks", record_merge)
        else:
            original_reduce = flox.groupby_reduce

            def record_configuration(*args: Any, **kwargs: Any) -> Any:
                sizes.append(dask.config.get("split_every"))
                return original_reduce(*args, **kwargs)

            monkeypatch.setattr(flox, "groupby_reduce", record_configuration)

        # Run with a different surrounding split_every setting
        source: Any = da.from_array(values, chunks=5) if execution == "dask" else values.copy()
        config = MultiprocConfig(chunks=5) if execution == "mp" else None
        with dask.config.set(split_every=7):
            result = gu.stats.stats(
                source,
                statistics,
                backend=backend,
                method="map-reduce",
                split_every=split_every,
                mp_config=config,
                **options,
            )
            assert dask.config.get("split_every") == 7

        # Check merge sizes, equality and loading
        expected_split = split_every
        if expected_split is None:
            expected_split = 8 if execution == "mp" else 7
        assert sizes and max(sizes) == expected_split
        assert all(1 <= size <= expected_split for size in sizes)
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        assert isinstance(source, da.Array) == (execution == "dask")
        np.testing.assert_array_equal(source.compute() if execution == "dask" else source, values)

    @pytest.mark.parametrize("method", ["blockwise", "cohorts"])
    def test_stats__flox_methods_chunk_invariance(
        self, method: Literal["auto", "map-reduce", "blockwise", "cohorts"]
    ) -> None:
        """Checks that Flox methods match in-memory statistics."""

        # Create groups crossing chunks
        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")

        values = np.arange(24, dtype=float)
        groups = np.repeat(np.arange(3), 8)
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1, 2, 3]}, "observed": False}
        statistics = ["mean", "sum", "min", "max"]
        expected = gu.stats.stats(values, statistics, **options)
        source = da.from_array(values, chunks=5)

        # Run the selected Flox method
        result = gu.stats.stats(source, statistics, backend="flox", method=method, **options)

        # Check equality and loading
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(source, da.Array)
        np.testing.assert_array_equal(source.compute(), values)

    def test_stats__flox_group_order(self) -> None:
        """Checks that Flox counts follow the requested group order."""

        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")

        # Create uneven groups in a different order from their labels
        values = np.arange(7.0)
        groups = np.array([0, 0, 1, 2, 0, 1, 0])
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [2, 0, 1, 3]}, "observed": False}
        statistics = ["mean", "totalcount", "percentagevalidpoints"]
        source = da.from_array(values, chunks=2)

        # Run default Flox reduction with a shorter final chunk
        expected = gu.stats.stats(values, statistics, **options)
        result = gu.stats.stats(source, statistics, backend="flox", **options)

        # Check counts, percentages and the unchanged lazy input
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        np.testing.assert_array_equal(result[("value", "totalcount")], [1, 4, 2, 0])
        assert isinstance(source, da.Array)
        np.testing.assert_array_equal(source.compute(), values)

    @pytest.mark.parametrize("method", ["auto", "cohorts"])
    def test_stats__flox_cohorts_grid(self, method: Literal["auto", "cohorts"]) -> None:
        """Checks that Flox reduces groups across rows and columns correctly."""

        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")

        # Create groups crossing chunks of 1 row and 2 columns
        values = np.arange(8.0).reshape(2, 4)
        groups = np.array([[1, 2, 2, 0], [0, 1, 2, 2]])
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1, 2]}}
        statistics = ["mean", "sum", "min", "max", "totalcount"]
        source = da.from_array(values, chunks=(1, 2))

        # Compare with complete groups in memory
        expected = gu.stats.stats(values, statistics, **options)
        result = gu.stats.stats(source, statistics, backend="flox", method=method, **options)

        # Check every group and the unchanged lazy input
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        assert result.loc[0, ("value", "mean")] == 3.5
        assert isinstance(source, da.Array)
        np.testing.assert_array_equal(source.compute(), values)

    @pytest.mark.parametrize("shape,grouped", [((24,), False), ((24,), True), ((6, 4), True)])
    def test_stats__flox_blockwise_exact_quantiles(self, shape: tuple[int, ...], grouped: bool) -> None:
        """Checks that Flox blockwise quantiles match exact results."""

        # Create complete groups with a NaN
        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")

        values = np.arange(24, dtype=float).reshape(shape)
        values.flat[3] = np.nan
        groups = np.repeat(np.arange(3), 8).reshape(shape)
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1, 2, 3]}} if grouped else {}
        statistics = ["median", "nmad", "90thpercentile", "iqr", "le90"]
        expected = gu.stats.stats(values, statistics, method="groupwise", **options)
        chunks = 5 if len(shape) == 1 else (2, 4)
        source = da.from_array(values, chunks=chunks)

        # Calculate blockwise statistics
        result = gu.stats.stats(source, statistics, backend="flox", method="blockwise", **options)

        # Check exact equality
        if grouped:
            pd.testing.assert_frame_equal(result, expected, check_exact=True)
        else:
            assert result == expected
        assert isinstance(source, da.Array)
        np.testing.assert_array_equal(source.compute(), values)

    def test_stats__groupwise_method(self) -> None:
        """Checks that groupwise preserves complete groups and value order."""

        # Create alternating groups with uneven chunks
        da = pytest.importorskip("dask.array")

        values = np.arange(14, dtype=float)
        groups = np.arange(14) % 2
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1]}}

        def first_value(array: Any) -> Any:
            """Return the first observation to check the order of each complete group."""

            return array[0]

        statistics: list[Any] = ["median", first_value]
        expected = gu.stats.stats(values, statistics, **options)
        source = da.from_array(values, chunks=5)

        # Run groupwise statistics
        result = gu.stats.stats(source, statistics, method="groupwise", **options)

        # Check equality and loading
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        assert result.attrs["grouped_stats"]["method"] == "groupwise"
        assert isinstance(source, da.Array)
        np.testing.assert_array_equal(source.compute(), values)

    @pytest.mark.parametrize("masked", [False, True])
    def test_stats__multiprocessing_summary(self, masked: bool, mp_cluster: MpCluster) -> None:
        """Checks that stats() combines ungrouped array tiles and optional masks in worker processes."""

        # Include nodata values and uneven edge tiles in both the array and raster routes

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
        config = MultiprocConfig(chunks=(2, 3), cluster=mp_cluster)
        array_summary = gu.stats.stats(values, mask=keep, mp_config=config)
        array_all = gu.stats.stats(values, "all", mask=keep, mp_config=config, method="groupwise")
        raster_summary = raster.stats(mask=keep, mp_config=config)
        reductions = raster.stats(["mean", "std", "sum", "validcount"], mask=keep, mp_config=config)

        # Match the in-memory values and preserve integer count results in both public APIs
        compare_dict(expected_summary, array_summary)
        compare_dict(expected_all, array_all)
        compare_dict(expected_summary, raster_summary)
        compare_dict(expected_reductions, reductions)
        assert isinstance(array_summary["Valid count"], int)
        assert isinstance(array_summary["Total count"], int)

    def test_stats__custom_summary_preserves_shape(self) -> None:
        """Checks that Multiproc custom summaries match in-memory results and see the complete array shape."""

        # Use a rectangular array whose full shape differs from every worker tile and from a flattened array
        values = np.arange(12, dtype=float).reshape(3, 4)
        config = MultiprocConfig(chunks=(2, 3))

        # Request the same custom summary eagerly and through Multiproc chunks
        expected_shape = gu.stats.stats(values, np.shape)
        expected_combined = gu.stats.stats(values, [np.shape, "mean"])
        shape = gu.stats.stats(values, np.shape, mp_config=config)
        combined = gu.stats.stats(values, [np.shape, "mean"], mp_config=config)

        # Preserve the complete shape and exactly match both in-memory output forms
        assert shape == expected_shape
        assert combined == expected_combined
        assert shape == combined["shape"] == values.shape
        assert combined["mean"] == pytest.approx(values.mean())

    @pytest.mark.parametrize("engine", ["numpy", "numba", "flox", "numbagg"])
    def test_stats__flox_engines(self, engine: str) -> None:
        """Checks that Flox engines match NumPy statistics."""

        if engine == "numba":
            pytest.importorskip("numba")

        # Create values and NumPy reference statistics
        pytest.importorskip("flox")
        if engine == "numbagg":
            pytest.importorskip("numbagg")
        da = pytest.importorskip("dask.array")

        values = np.arange(41, dtype=float)
        source = da.from_array(values, chunks=7)
        expected = {
            "mean": np.mean(values),
            "std": np.std(values),
            "sum": np.sum(values),
            "min": np.min(values),
            "max": np.max(values),
            "rmse": np.sqrt(np.mean(values**2)),
        }

        # Calculate statistics with the selected Flox engine
        result = gu.stats.stats(
            source,
            list(expected),
            backend="flox",
            flox_engine=engine,
        )

        # Allow rounding from summing chunks
        assert isinstance(source, da.Array)
        for name, reference in expected.items():
            np.testing.assert_allclose(result[name], reference)
        np.testing.assert_array_equal(source.compute(), values)

    @pytest.mark.parametrize(
        "backend,execution,method",
        [("geoutils", execution, method) for execution in ("dask", "mp") for method in ("groupwise", "map-reduce")]
        + [("flox", "dask", "map-reduce")],
    )
    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__parallel_distribution(
        self,
        backend: str,
        execution: str,
        grouped: bool,
        method: Literal["auto", "groupwise", "map-reduce"],
    ) -> None:
        """Checks that chunked distribution statistics match in-memory results."""

        # Create symmetric values whose chunk medians differ
        if method != "groupwise":
            pytest.importorskip("crick")
        if backend == "flox":
            pytest.importorskip("flox")
        if execution == "dask":
            dask = pytest.importorskip("dask")
            da = pytest.importorskip("dask.array")
            from dask.callbacks import Callback

        values = np.tile(np.arange(-20, 21, dtype=float), 17)
        groups = np.tile(np.arange(41) % 3, 17)
        statistics = ["mean", "std", "sum", "median", "nmad", "90thpercentile", "iqr", "le90"]
        options: dict[str, Any] = {"backend": backend, "method": method}
        if method != "groupwise":
            options["tdigest_compression"] = 200
        if grouped:
            options.update(by={"zone": groups}, categories={"zone": [0, 1, 2, 3]}, observed=False)
        expected = gu.stats.stats(values, statistics, **options)
        source: Any = da.from_array(values, chunks=37) if execution == "dask" else values.copy()

        def check_task(key: Any, result: Any, *args: Any) -> None:
            """Check that no task collects the complete observation vector."""

            if method != "groupwise" and isinstance(result, np.ndarray):
                assert result.shape != values.shape

        # Calculate chunked statistics
        scheduler = dask.config.set(scheduler="synchronous") if execution == "dask" else nullcontext()
        callback = Callback(posttask=check_task) if execution == "dask" else nullcontext()
        with scheduler, callback:
            config = MultiprocConfig(chunks=37) if execution == "mp" else None
            result = gu.stats.stats(source, statistics, **options, mp_config=config)

        # Allow rounding and small t-digest errors
        if grouped:
            pd.testing.assert_frame_equal(result, expected)
        else:
            for name in statistics:
                if method != "groupwise" and name in {"median", "nmad", "90thpercentile", "iqr", "le90"}:
                    np.testing.assert_allclose(result[name], expected[name], atol=0.1)
                else:
                    assert result[name] == pytest.approx(expected[name])
            assert abs(result["median"] - np.median(values)) < 0.1
            assert abs(result["nmad"] - 1.4826 * np.median(np.abs(values - np.median(values)))) < 0.1
        assert is_dask_array(source) == (execution == "dask")
        if is_dask_array(source):
            np.testing.assert_array_equal(source.compute(), values)
        else:
            np.testing.assert_array_equal(source, values)

    @pytest.mark.parametrize("backend,execution", [("geoutils", "dask"), ("geoutils", "mp"), ("flox", "dask")])
    def test_stats__masked_nan(self, backend: Literal["geoutils", "flox"], execution: str) -> None:
        """Checks that an exposed NaN is missing in counts and estimates."""

        pytest.importorskip("crick")
        if backend == "flox":
            pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")

        # Create an exposed NaN and a separate masked value
        values = np.ma.array([1.0, np.nan, 3.0, 99.0], mask=[False, False, False, True])
        source = da.from_array(values, chunks=3) if execution == "dask" else values.copy()
        config = MultiprocConfig(chunks=3) if execution == "mp" else None
        statistics = ["mean", "median", "sum", "validcount"]

        # Reduce uneven chunks and compare with the same backend in memory
        expected = gu.stats.stats(values, statistics, backend=backend)
        result = gu.stats.stats(source, statistics, backend=backend, mp_config=config)

        # Check estimates, counts and the original mask
        assert result == expected == {"mean": 2.0, "median": 2.0, "sum": 4.0, "validcount": 2}
        assert isinstance(source, da.Array) == (execution == "dask")
        computed = source.compute() if execution == "dask" else source
        np.testing.assert_array_equal(computed, values)
        np.testing.assert_array_equal(np.ma.getmaskarray(values), [False, False, False, True])

    @pytest.mark.parametrize("backend,execution", [("geoutils", "dask"), ("geoutils", "mp"), ("flox", "dask")])
    def test_stats__global_infinity_chunk(self, backend: Literal["geoutils", "flox"], execution: str) -> None:
        """Checks edge case of global statistics works in chunks."""

        # Create values with infinity and NaN
        if backend == "flox":
            pytest.importorskip("flox")
        if execution == "dask":
            da = pytest.importorskip("dask.array")

        values = np.array([1.0, np.inf, np.nan, 3.0, 5.0])
        statistics = ["mean", "sum", "min", "max", "validcount"]
        expected = gu.stats.stats(values, statistics)
        source = da.from_array(values, chunks=2) if execution == "dask" else values.copy()
        config = MultiprocConfig(chunks=2) if execution == "mp" else None

        # Calculate chunked statistics
        result = gu.stats.stats(source, statistics, backend=backend, mp_config=config)

        # Check equality and loading
        assert result == expected
        assert is_dask_array(source) == (execution == "dask")

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__flox_reduction_chunk(self, grouped: bool, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that chunked Flox statistics run independently of GeoUtils reductions."""

        # Split constant groups across multiple chunks
        pytest.importorskip("flox")
        pytest.importorskip("crick")
        da = pytest.importorskip("dask.array")

        from geoutils.stats import reduction

        groups = np.arange(41) % 2
        values = np.where(groups, 7.0, 2.0) if grouped else np.full(41, 2.0)
        values[3] = np.nan
        eager = {"height": values, "code": 2**60 + groups if grouped else np.full(41, 2**60, dtype=np.int64)}
        source = {name: da.from_array(array, chunks=7) for name, array in eager.items()}
        options: dict[str, Any] = {"by": {"zone": groups}, "categories": {"zone": [0, 1, 2]}} if grouped else {}
        statistics = ["mean", "std", "median", "nmad", "iqr", "rmse", "min", "max"]
        expected = gu.stats.stats(eager, statistics, **options)

        # Fail if the native reducer is called through the Flox backend
        def fail_native(*args: Any, **kwargs: Any) -> None:
            pytest.fail("Flox must perform its own reductions and merges")

        for name in ("_reduce_block", "_merge_blocks", "_aggregate_complete_groups"):
            monkeypatch.setattr(reduction, name, fail_native)
        result = gu.stats.stats(source, statistics, backend="flox", method="map-reduce", **options)

        # Check equality and loading
        assert all(isinstance(array, da.Array) for array in source.values())
        if grouped:
            pd.testing.assert_frame_equal(result, expected)
        else:
            for value_name in eager:
                for name in statistics:
                    np.testing.assert_allclose(result[value_name][name], expected[value_name][name])
        for name in eager:
            np.testing.assert_array_equal(source[name].compute(), eager[name])


class TestStatsBackendErrors:
    """Test module for errors/warnings in stats()."""

    @pytest.mark.parametrize(
        "options,message,error",
        [
            ({"method": "unknown"}, "Argument ``method``", ValueError),
            ({"reindex": 1}, "Argument ``reindex``", TypeError),
            ({"split_every": True}, "Argument ``split_every``", TypeError),
            ({"split_every": 2.5}, "Argument ``split_every``", TypeError),
            ({"split_every": 1}, "Argument ``split_every``", ValueError),
            ({"split_every": 0}, "Argument ``split_every``", ValueError),
            ({"method": "groupwise", "split_every": 2}, "applies to group summaries", ValueError),
            ({"method": "blockwise"}, "requires backend='flox'", ValueError),
            ({"method": "cohorts"}, "requires backend='flox'", ValueError),
            ({"backend": "flox", "method": "groupwise"}, "requires backend='geoutils'", ValueError),
            ({"method": "groupwise", "reindex": False}, "applies to group summaries", ValueError),
            ({"backend": "flox", "method": "cohorts", "reindex": True}, "requires reindex=False", ValueError),
        ],
    )
    def test_stats__error_method_options(self, options: dict[str, Any], message: str, error: type[Exception]) -> None:
        """Checks an error is raised for invalid method options."""

        # Require Flox only for options that select its backend
        if options.get("backend") == "flox":
            pytest.importorskip("flox")

        # Check the error
        with pytest.raises(error, match=message):
            gu.stats.stats(np.arange(6.0), "mean", **options)

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__error_flox_multiproc(self, grouped: bool) -> None:
        """Checks an error is raised for Flox multiprocessing."""

        pytest.importorskip("flox")

        # Create values and optional groups
        values = np.arange(7, dtype=float)
        options: dict[str, Any] = {"by": {"zone": np.arange(7) % 2}, "categories": {"zone": [0, 1]}} if grouped else {}

        # Check the error before starting workers
        with pytest.raises(ValueError, match="Flox backend does not support multiprocessing"):
            gu.stats.stats(values, "mean", backend="flox", mp_config=MultiprocConfig(chunks=3), **options)
