"""Tests for statistics grouped by continuous and categorical variables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

import geoutils as gu
from geoutils.raster.xr_accessor import RasterAccessor


def test_grouped_stats_preserves_intervals_counts_and_masks() -> None:
    """Group masks retain membership independently of missing selected values."""

    values = {
        "first": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0]),
        "second": np.arange(6, dtype=float),
    }
    grouper = np.arange(6, dtype=float)
    user_mask = np.array([True, False, True, True, True, True])

    # Request masks with two explicit intervals whose final edge includes the maximum
    table, masks = gu.stats.grouped_stats(
        values,
        {"slope": grouper},
        bins={"slope": [0, 3, 6]},
        mask=user_mask,
        statistics=["median"],
        return_masks=True,
    )

    assert isinstance(table.index, pd.IntervalIndex)
    assert list(table.columns.names) == ["value", "statistic"]
    assert table[("first", "count")].tolist() == [1, 3]
    assert table[("second", "count")].tolist() == [2, 3]
    assert isinstance(masks, Mapping)
    assert list(masks) == list(table.index)

    # Check that group masks partition all eligible locations without applying value validity
    group_masks = [np.asarray(masks[key]) for key in masks]
    assert [int(np.count_nonzero(group_mask)) for group_mask in group_masks] == [2, 3]
    assert np.array_equal(np.logical_or.reduce(group_masks), user_mask)
    assert not np.any(group_masks[0] & group_masks[1])


def test_grouped_stats_combines_categories_and_empty_groups() -> None:
    """A MultiIndex retains ordered interval and category levels including empty combinations."""

    values = np.arange(6, dtype=float)
    continuous = np.array([0, 0, 1, 1, 2, 2], dtype=float)
    categorical = np.array(["forest", "forest", "forest", "ice", "ice", "ice"])

    # Retain the complete declared product to make empty groups explicit
    table = gu.stats.grouped_stats(
        values,
        {"elevation": continuous, "surface": categorical},
        bins={"elevation": [0, 1, 2, 3]},
        categories={"surface": ["forest", "ice"]},
        statistics="mean",
        observed=False,
    )

    assert isinstance(table.index, pd.MultiIndex)
    assert isinstance(table.index.levels[0], pd.IntervalIndex)
    assert isinstance(table.index.levels[1], pd.CategoricalIndex)
    assert table.index.levels[1].ordered
    assert len(table) == 6
    assert table.loc[(pd.Interval(0, 1, closed="left"), "ice"), ("value", "count")] == 0
    assert np.isnan(table.loc[(pd.Interval(0, 1, closed="left"), "ice"), ("value", "mean")])


def test_grouped_stats_respects_interval_closure_and_nonfinite_values() -> None:
    """Exact intervals and complete statistics consistently exclude nonfinite selected values."""

    intervals = pd.IntervalIndex.from_breaks([0, 1, 2], closed="right", name="distance")
    table, masks = gu.stats.grouped_stats(
        np.array([100.0, 1.0, np.inf]),
        {"distance": np.array([0.0, 1.0, 2.0])},
        bins={"distance": intervals},
        statistics="all",
        return_masks=True,
    )

    assert table.index.equals(intervals)
    assert table[("value", "count")].tolist() == [1, 0]
    assert table[("value", "totalcount")].tolist() == [1, 1]
    assert table.loc[intervals[0], ("value", "mean")] == 1
    assert np.isnan(table.loc[intervals[1], ("value", "mean")])
    assert not np.asarray(masks[intervals[0]])[0]


def test_grouped_stats_subsampling_does_not_change_masks() -> None:
    """Returned masks describe complete groups while statistics use the requested sample."""

    values = np.arange(20, dtype=float)
    groups = np.arange(20, dtype=float)
    table, masks = gu.stats.grouped_stats(
        values,
        {"distance": groups},
        bins={"distance": [0, 10, 20]},
        statistics="mean",
        subsample=6,
        random_state=42,
        return_masks=True,
    )

    assert int(table[("value", "count")].sum()) == 6
    assert sum(int(np.count_nonzero(masks[key])) for key in masks) == 20


def test_grouped_stats_dask_matches_numpy_and_keeps_masks_lazy() -> None:
    """Dask reductions match eager summaries without materializing returned group masks."""

    da = pytest.importorskip("dask.array")
    values = np.arange(12, dtype=float).reshape(3, 4)
    grouper = np.arange(12, dtype=float).reshape(3, 4)
    expected = gu.stats.grouped_stats(values, {"x": grouper}, bins={"x": [0, 4, 8, 12]}, statistics="mean")

    # Use different chunk layouts to exercise alignment in the shared group layer
    table, masks = gu.stats.grouped_stats(
        da.from_array(values, chunks=(2, 2)),
        {"x": da.from_array(grouper, chunks=(1, 4))},
        bins={"x": [0, 4, 8, 12]},
        statistics="mean",
        return_masks=True,
    )

    pd.testing.assert_frame_equal(table, expected)
    first_mask = masks[next(iter(masks))]
    assert isinstance(first_mask, da.Array)
    assert int(first_mask.sum().compute()) == 4


def test_raster_grouped_stats_returns_writable_raster_masks(tmp_path: Path) -> None:
    """Raster group masks retain their grid and Boolean mask type through file output."""

    transform = Affine(10, 0, 100, 0, -10, 200)
    raster = gu.Raster.from_array(np.arange(1, 7, dtype=float).reshape(2, 3), transform, 32631)
    grouper = np.arange(6, dtype=float).reshape(2, 3)
    table, masks = raster.grouped_stats(
        {"slope": grouper},
        bins={"slope": [0, 3, 6]},
        statistics="mean",
        return_masks=True,
    )

    first_mask = masks[table.index[0]]
    assert isinstance(first_mask, gu.Raster)
    assert first_mask.is_mask
    assert first_mask.georeferenced_grid_equal(raster)
    output_path = tmp_path / "group_mask.tif"
    first_mask.to_file(output_path)
    reopened = gu.Raster(output_path, is_mask=True, load_data=True)
    assert reopened.is_mask
    assert np.array_equal(reopened.data, first_mask.data)


def test_xarray_and_pointcloud_grouped_stats_preserve_support_types() -> None:
    """Accessor and point cloud methods return masks matching their native support interfaces."""

    transform = Affine(1, 0, 0, 0, -1, 2)
    raster = RasterAccessor.from_array(np.arange(6, dtype=float).reshape(2, 3), transform, 32631)
    raster_table, raster_masks = raster.rst.grouped_stats(
        {"x": np.arange(6, dtype=float).reshape(2, 3)},
        bins={"x": 2},
        statistics="mean",
        return_masks=True,
    )
    raster_mask = raster_masks[raster_table.index[0]]
    assert raster_mask.dtype == bool
    assert raster_mask.rst.is_mask
    assert raster_mask.rst.georeferenced_grid_equal(raster)

    # Select a native point column as a categorical grouper
    pointcloud = gu.PointCloud.from_xyz(
        x=np.arange(6),
        y=np.zeros(6),
        z=np.arange(1, 7, dtype=float),
        crs=32631,
    )
    pointcloud.ds["surface"] = ["land", "land", "ice", "ice", "land", "ice"]
    point_table, point_masks = pointcloud.grouped_stats(
        {"surface": "surface"},
        categories={"surface": ["land", "ice"]},
        statistics="mean",
        return_masks=True,
    )
    point_mask = point_masks[point_table.index[0]]
    assert isinstance(point_mask, gu.PointCloud)
    assert point_mask.is_mask
    assert point_mask.georeferenced_coords_equal(pointcloud)

    # Add a Boolean column when the source values are stored as numeric point elevations
    elevation_pointcloud = gu.PointCloud.from_xyz(
        x=np.arange(4),
        y=np.zeros(4),
        z=np.arange(4, dtype=float),
        crs=32631,
        use_z=True,
    )
    elevation_table, elevation_masks = elevation_pointcloud.grouped_stats(
        {"surface": np.array(["land", "land", "ice", "ice"])},
        categories={"surface": ["land", "ice"]},
        statistics="mean",
        return_masks=True,
    )
    elevation_mask = elevation_masks[elevation_table.index[0]]
    assert isinstance(elevation_mask, gu.PointCloud)
    assert elevation_mask.is_mask
    assert elevation_mask.data_column == "group_mask"
    assert elevation_mask.georeferenced_coords_equal(elevation_pointcloud)


def test_raster_grouped_stats_accepts_vector_feature_categories() -> None:
    """Vector feature values become ordered categories on raster support."""

    raster = gu.Raster.from_array(
        np.arange(1, 5, dtype=float).reshape(2, 2),
        Affine(1, 0, 0, 0, -1, 2),
        32631,
    )
    zones = gu.Vector(
        gpd.GeoDataFrame(
            {"zone": ["west", "east"]},
            geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2)],
            crs=32631,
        )
    )

    table = raster.grouped_stats(
        {"zone": (zones, "zone")},
        statistics="mean",
    )

    assert isinstance(table.index, pd.CategoricalIndex)
    assert table[("band_1", "count")].tolist() == [2, 2]
    assert table[("band_1", "mean")].tolist() == [2.0, 3.0]


def test_plot_grouped_stats_supports_one_and_two_dimensions() -> None:
    """The plotting helper creates count panels for both supported layouts."""

    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    one_dimensional = gu.stats.grouped_stats(
        np.arange(6, dtype=float),
        {"x": np.arange(6, dtype=float)},
        bins={"x": [0, 3, 6]},
        statistics="mean",
    )
    axes_1d = gu.stats.plot_grouped_stats(one_dimensional, statistic="mean")
    assert set(axes_1d) == {"count", "statistic"}

    two_dimensional = gu.stats.grouped_stats(
        np.arange(6, dtype=float),
        {"x": np.array([0, 0, 1, 1, 2, 2]), "surface": np.array(["a", "b", "a", "b", "a", "b"])},
        bins={"x": [0, 1, 2, 3]},
        categories={"surface": ["a", "b"]},
        statistics="mean",
    )
    axes_2d = gu.stats.plot_grouped_stats(two_dimensional, statistic="mean")
    assert set(axes_2d) == {"count_x", "count_y", "statistic", "colorbar"}
    plt.close("all")


@pytest.mark.parametrize("kind", ["integer", "boolean", "string"])
def test_grouped_stats_preserves_masked_values_and_categories(kind: str) -> None:
    """Masked integer values and masked category labels must keep independent validity."""

    # Mask a value and a different group label to distinguish counts from group membership
    values = np.ma.array([1, 2, 3, 4, 5, 6], mask=[False, True, False, False, False, False])
    group_values = {
        "integer": [0, 0, 0, 1, 1, 1],
        "boolean": [False, False, False, True, True, True],
        "string": ["a", "a", "a", "b", "b", "b"],
    }
    groups = np.ma.array(group_values[kind], mask=[False, False, True, False, False, False])
    categories = {"integer": {"group": [0, 1]}, "boolean": None, "string": {"group": ["a", "b", "N/A"]}}

    table, masks = gu.stats.grouped_stats(
        values,
        {"group": groups},
        categories=categories[kind],
        statistics="mean",
        return_masks=True,
    )

    assert table[("value", "count")].tolist() == [1, 3]
    assert table[("value", "mean")].tolist() == [1, 5]
    assert [int(np.count_nonzero(masks[key])) for key in masks] == [2, 3]


def test_raster_grouped_stats_excludes_masked_integer_data_and_boolean_mask() -> None:
    """Raster statistics must retain value gaps separately from masked group support."""

    data = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [False, False, False]])
    raster = gu.Raster.from_array(data, Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
    mask = raster.from_array(
        np.ma.array([[True, True, True], [True, False, True]], mask=[[False, False, True], [False, False, False]]),
        raster.transform,
        raster.crs,
    )

    table, masks = raster.grouped_stats(
        {"group": np.zeros(data.shape, dtype=int)},
        categories={"group": [0]},
        statistics="mean",
        mask=mask,
        return_masks=True,
    )

    assert table[("band_1", "count")].tolist() == [3]
    assert table[("band_1", "mean")].tolist() == [pytest.approx(11 / 3)]
    assert int(np.count_nonzero(masks[0].data)) == 4
