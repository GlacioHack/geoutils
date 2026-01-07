from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from pandas.testing import assert_frame_equal

from geoutils import Raster
from geoutils.stats import grouped_stats


@pytest.fixture
def transform():  # type: ignore
    return (30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)


@pytest.fixture
def slope_raster(transform):  # type: ignore
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    return Raster.from_array(data, transform, 32645)


@pytest.fixture
def elev_raster(transform):  # type: ignore
    data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])
    return Raster.from_array(data, transform, 32645)


@pytest.fixture
def mask_raster(tmp_path, transform):  # type: ignore
    data_mask = np.array(
        [
            [True, False, True, False, False],
            [True, False, True, False, False],
            [True, True, False, False, False],
            [True, True, False, False, False],
        ]
    )
    mask = Raster.from_array(data_mask, transform, 32645)
    mask.to_file(tmp_path / "mask_test.tif")
    return Raster(tmp_path / "mask_test.tif", is_mask=True)


@pytest.fixture
def segm_raster(transform):  # type: ignore
    data_segm = np.array([[1, 1, 2, 3, 1], [2, 2, 3, 1, 1], [1, 2, 1, 1, 3], [2, 2, 2, 3, 3]])
    return Raster.from_array(data_segm, transform, 32645)


@pytest.fixture
def statistics():  # type: ignore
    return ["mean", "min"]


@pytest.fixture
def aggregated_vars(elev_raster):  # type: ignore
    return {"raster": elev_raster}


def test_init_init_binnings_attributes(elev_raster: Raster, transform: List[Any]) -> None:

    crs, shape, transform_test = grouped_stats.init_binnings_attributes(elev_raster)

    assert crs == 32645
    assert shape == (4, 5)
    assert transform_test == Affine(30.0, 0.0, 478000.0, 0.0, -30.0, 3108140.0)


def test_from_raster_to_flattened(elev_raster: Raster) -> None:
    dict_test = {"elev_test": elev_raster}
    array_dict_test = grouped_stats.from_raster_to_flattened(dict_test)
    assert array_dict_test["elev_test"].shape == (20,)


def test_is_interval() -> None:
    test_interval = [0, 5, 20, 40, np.inf]
    assert grouped_stats.is_interval(test_interval) is True


@pytest.mark.parametrize(
    "interval, type_error, match",
    [
        (
            [[0, 5, 20, 40, np.inf], [0, 5, 20, 40, np.inf]],
            ValueError,
            "If bins is an interval, it must be a 1-dimensional array",
        ),
        ([0, "a", 20, 40, np.inf], TypeError, "Bins must be a List[Any] of number"),
        ([0], ValueError, "Bins must be of size > 2"),
        ([0, 5, -2, 40, np.inf], ValueError, "Values must be strictly increasing."),
    ],
)  # type: ignore
def test_is_interval_with_error(interval, type_error, match) -> None:

    with pytest.raises(type_error, match=match):
        _ = grouped_stats.is_interval(interval)


@pytest.mark.parametrize(
    "bins_test",
    [
        ({"slope1": [0, 1, 2]}),  # different size of group_by
        ({"slope3": [0, 1, 2], "slope4": [0, 1, 2]}),  # different keys
    ],
)  # type: ignore
def test_grouped_stats_errors(slope_raster, aggregated_vars, bins_test, statistics) -> None:
    group_by_test = {"slope1": slope_raster, "slope2": slope_raster}
    with pytest.raises(ValueError, match="One bins/mask/segmentation entry per input array required."):
        _ = grouped_stats.grouped_stats(group_by_test, bins_test, aggregated_vars, statistics)


def test_grouped_stats_interval(slope_raster: Raster, aggregated_vars: Dict[str, Any], statistics: List[Any]) -> None:
    group_by_test = {"slope1": slope_raster}
    expected_df = pd.DataFrame(
        {
            ("raster", "mean"): [3, 8, 15.5],
            ("raster", "min"): [1, 6, 11],
        },
        index=pd.CategoricalIndex(
            pd.IntervalIndex.from_breaks([-0.001, 5.0, 10.0, np.inf], closed="right"),
            ordered=True,
            name="groupby_slope1",
        ),
    )
    df_test = grouped_stats.grouped_stats(group_by_test, {"slope1": [0, 5, 10, np.inf]}, aggregated_vars, statistics)
    assert_frame_equal(df_test, expected_df)


def test_grouped_stats_mask(
    slope_raster: Raster, aggregated_vars: Dict[str, Any], mask_raster: Raster, statistics: List[Any]  # type: ignore
) -> None:
    group_by_test = {"slope1": slope_raster}
    expected_df = pd.DataFrame(
        {
            ("raster", "mean"): [11.333333, 9.25],
            ("raster", "min"): [2, 1],
        },
        index=pd.CategoricalIndex(
            [False, True], categories=[False, True], ordered=False, dtype="category", name="groupby_slope1"
        ),
    )

    df_test = grouped_stats.grouped_stats(group_by_test, {"slope1": mask_raster}, aggregated_vars, statistics)
    assert_frame_equal(df_test, expected_df)


def test_grouped_stats_segm(
    slope_raster: Raster, aggregated_vars: Dict[str, Any], segm_raster: Raster, statistics: List[Any]
) -> None:
    group_by_test = {"slope1": slope_raster}

    expected_df = pd.DataFrame(
        {
            ("raster", "mean"): [8.125000, 11.285714, 13.200000],
            ("raster", "min"): [1, 3, 4],
        },
        index=pd.CategoricalIndex(
            [1, 2, 3], categories=[1, 2, 3], ordered=False, dtype="category", name="groupby_slope1"
        ),
    )
    df_test = grouped_stats.grouped_stats(group_by_test, {"slope1": segm_raster}, aggregated_vars, statistics)
    assert_frame_equal(df_test, expected_df)


def test_intersection_stats(
    slope_raster: Raster, aggregated_vars: Dict[str, Any], segm_raster: Raster, statistics: List[Any]
) -> None:
    group_by_test = {"slope1": slope_raster, "slope2": slope_raster}
    bins_test = {"slope1": [0, 5, 10, np.inf], "slope2": segm_raster}

    df_test = grouped_stats.grouped_stats(group_by_test, bins_test, aggregated_vars, statistics)

    assert df_test.loc[(pd.Interval(10.0, np.inf), 3), ("raster", "mean")] == (15 + 19 + 20) / 3
    assert df_test.loc[(pd.Interval(10.0, np.inf), 3), ("raster", "min")] == 15

    assert df_test.loc[(pd.Interval(-0.001, 5), 1), ("raster", "mean")] == (1 + 2 + 5) / 3
    assert df_test.loc[(pd.Interval(-0.001, 5), 1), ("raster", "min")] == 1

    assert df_test.loc[(pd.Interval(10.0, np.inf), 2), ("raster", "mean")] == (16 + 12 + 17 + 18) / 4
    assert df_test.loc[(pd.Interval(10.0, np.inf), 2), ("raster", "min")] == 12
