"""Tests for grouping statistics by continuous and categorical variables."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from shapely.geometry import box

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.multiproc import MultiprocConfig


class TestGroupedStats:
    """
    Test module for grouped statistics, i.e. stats(by=), with eager inputs.

    Dask and Multiproc inputs are covered in TestGroupedStatsChunked further below.

    The tests cover group definitions, sampling, vector zones, returned tables and masks, plotting, and using the
    optional Flox backend.
    """

    def test_stats__interval_counts_and_masks(self) -> None:
        """
        Checks that group masks include selected locations with nodata values while counts exclude them.

        Contrary to cosample() that wants to sample the common valid locations, for stats we want to keep
        nodata as it is valuable information for the user, and is described directly by valid/total counts.

        """

        # Create two values with different nodata patterns and a user mask with excluded points
        values = {
            "first": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0]),
            "second": np.arange(6, dtype=float),
        }
        grouper = np.arange(6, dtype=float)
        user_mask = np.array([True, False, True, True, True, True])

        # Group values into two explicit intervals, and return their final mask
        table, masks = gu.stats.stats(
            values,
            "median",
            by={"slope": grouper},
            bins={"slope": [0, 3, 6]},
            mask=user_mask,
            return_masks=True,
        )

        # Check interval labels, separate value counts, and returned mask keys
        assert isinstance(table.index, pd.IntervalIndex)
        assert list(table.columns.names) == ["value", "statistic"]
        assert table[("first", "count")].tolist() == [1, 3]
        assert table[("second", "count")].tolist() == [2, 3]
        assert isinstance(masks, Mapping)
        assert list(masks) == list(table.index)

        # Check that masks split locations allowed by the user mask without removing selected nodata values
        group_masks = [np.asarray(masks[key]) for key in masks]
        assert [int(np.count_nonzero(group_mask)) for group_mask in group_masks] == [2, 3]
        assert np.array_equal(np.logical_or.reduce(group_masks), user_mask)
        assert not np.any(group_masks[0] & group_masks[1])

    def test_stats__combines_categories_and_empty_groups(self) -> None:
        """Checks that two grouping variables return labels in the input order, and include empty combinations."""

        # Create numeric bins and named categories with one combination absent in the data
        values = np.arange(6, dtype=float)
        continuous = np.array([0, 0, 1, 1, 2, 2], dtype=float)
        categorical = np.array(["forest", "forest", "forest", "ice", "ice", "ice"])

        # Request every defined category and bin combination, including the empty one
        table = gu.stats.stats(
            values,
            by={"elevation": continuous, "surface": categorical},
            bins={"elevation": [0, 1, 2, 3]},
            categories={"surface": ["forest", "ice"]},
            statistics="mean",
            observed=False,
        )

        # Check index types, order, and the empty group's zero count and NaN mean
        assert isinstance(table.index, pd.MultiIndex)
        assert isinstance(table.index.levels[0], pd.IntervalIndex)
        assert isinstance(table.index.levels[1], pd.CategoricalIndex)
        assert table.index.levels[1].ordered
        assert len(table) == 6
        assert table.loc[(pd.Interval(0, 1, closed="left"), "ice"), ("value", "count")] == 0
        assert np.isnan(table.loc[(pd.Interval(0, 1, closed="left"), "ice"), ("value", "mean")])

    def test_stats__respects_interval_closure_and_nonfinite_values(self) -> None:
        """Checks that right-closed intervals and selected nodata values affect groups and counts separately."""

        # Define right-closed intervals with one value below all intervals and one infinite selected value
        intervals = pd.IntervalIndex.from_breaks([0, 1, 2], closed="right", name="distance")

        # Calculate all statistics and return the complete interval masks
        table, masks = gu.stats.stats(
            np.array([100.0, 1.0, np.inf]),
            by={"distance": np.array([0.0, 1.0, 2.0])},
            bins={"distance": intervals},
            statistics="all",
            return_masks=True,
        )

        # Check the interval edge rules, finite counts, total counts, means, and locations in each group mask
        assert table.index.equals(intervals)
        assert table[("value", "count")].tolist() == [1, 0]
        assert table[("value", "totalcount")].tolist() == [1, 1]
        assert table.loc[intervals[0], ("value", "mean")] == 1
        assert np.isnan(table.loc[intervals[1], ("value", "mean")])
        assert not np.asarray(masks[intervals[0]])[0]

    def test_stats__subsampling_does_not_change_masks(self) -> None:
        """Checks that subsampling limits statistic counts without modifying returned group masks."""

        # Split twenty values into two intervals
        values = np.arange(20, dtype=float)
        groups = np.arange(20, dtype=float)
        # Calculate statistics from six sampled locations and request complete masks
        table, masks = gu.stats.stats(
            values,
            by={"distance": groups},
            bins={"distance": [0, 10, 20]},
            statistics="mean",
            subsample=6,
            random_state=42,
            return_masks=True,
        )

        # Check the sampled statistic count and the full twenty-location mask count
        assert int(table[("value", "count")].sum()) == 6
        assert sum(int(np.count_nonzero(masks[key])) for key in masks) == 20

    def test_stats__raster_mask_type(self, tmp_path: Path) -> None:
        """Checks that raster group masks have the same grid and boolean type when written and reopened."""

        # Create a georeferenced raster and split its cells into two numeric intervals
        transform = Affine(10, 0, 100, 0, -10, 200)
        raster = gu.Raster.from_array(np.arange(1, 7, dtype=float).reshape(2, 3), transform, 32631)
        grouper = np.arange(6, dtype=float).reshape(2, 3)
        table, masks = raster.stats(
            "mean",
            by={"slope": grouper},
            bins={"slope": [0, 3, 6]},
            return_masks=True,
        )

        # Check that the first returned mask is a boolean Raster on the source grid
        first_mask = masks[table.index[0]]
        assert isinstance(first_mask, gu.Raster)
        assert first_mask.is_mask
        assert first_mask.georeferenced_grid_equal(raster)
        # Write the first group mask to disk, reopen it, and compare its type and values
        output_path = tmp_path / "group_mask.tif"
        first_mask.to_file(output_path)
        reopened = gu.Raster(output_path, is_mask=True, load_data=True)
        assert reopened.is_mask
        assert np.array_equal(reopened.data, first_mask.data)

    @pytest.mark.parametrize("source_type", ["raster", "xarray", "pointcloud", "dataframe"])
    def test_stats__mask_types(self, source_type: str) -> None:
        """Checks that every eager spatial input returns a boolean mask on its common support."""

        # Create the same six values and two groups as either raster cells or points
        values = np.arange(1, 7, dtype=float)
        groups = np.arange(6) % 2
        if source_type in {"raster", "xarray"}:
            values = values.reshape(2, 3)
            groups = groups.reshape(2, 3)
            raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 2), 32631)
            source: Any = raster if source_type == "raster" else raster.to_xarray().rst
        else:
            pointcloud = gu.PointCloud.from_xyz(np.arange(6), np.zeros(6), values, crs=32631)
            source = pointcloud if source_type == "pointcloud" else pointcloud.ds.pc

        # Request the complete mask for each group
        table, masks = source.stats("mean", by={"zone": groups}, categories={"zone": [0, 1]}, return_masks=True)
        mask = masks[table.index[0]]

        # Check that the mask has the matching spatial type, common support, and group values
        if source_type == "raster":
            assert isinstance(mask, gu.Raster)
            assert mask.is_mask
            assert mask.georeferenced_grid_equal(raster)
            mask_values = mask.data
        elif source_type == "xarray":
            assert mask.rst.is_mask
            assert mask.rst.georeferenced_grid_equal(raster)
            mask_values = mask.data
        else:
            assert isinstance(mask, gu.PointCloud if source_type == "pointcloud" else gpd.GeoDataFrame)
            interface = mask if source_type == "pointcloud" else mask.pc
            assert interface.is_mask
            assert interface.georeferenced_coords_equal(pointcloud)
            mask_values = interface.data
        assert np.array_equal(np.asarray(mask_values).reshape(groups.shape), groups == 0)

    @pytest.mark.parametrize(
        "by,bins,categories,message",
        [
            ({}, None, None, "at least one named grouper"),
            ({"zone": np.zeros((2, 2))}, {"missing": 2}, None, "do not match"),
            ({"zone": np.zeros((2, 2))}, {"zone": 2}, {"zone": [0]}, "cannot define both"),
            ({"zone": np.zeros((2, 2))}, {"zone": [0, 0, 1]}, None, "strictly increasing"),
            ({"zone": np.zeros((2, 2))}, None, {"zone": [0, 0]}, "unique"),
        ],
    )
    def test_stats__error_invalid_group_definitions(
        self, by: Any, bins: Any, categories: Any, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that invalid group definitions fail before an unloaded raster reads any data."""

        # Write a raster file to disk with four valid cells, then make any attempt to load all cells fail
        raster = gu.Raster.from_array(np.ones((2, 2)), Affine(1, 0, 0, 0, -1, 2), 32631)
        path = tmp_path / "group_validation.tif"
        raster.to_file(path)
        source = gu.Raster(path)

        def fail_load(*args: Any, **kwargs: Any) -> None:
            """Reject data access before group definition validation finishes."""
            raise AssertionError("Group definitions must be checked before reading values.")

        monkeypatch.setattr(source, "load", fail_load)

        # Raise the group definition error without consulting the source's values
        with pytest.raises(ValueError, match=message):
            source.stats("mean", by=by, bins=bins, categories=categories)
        assert not source.is_loaded

    @pytest.mark.parametrize("source_type", ["raster", "xarray", "pointcloud", "geopandas"])
    def test_stats__subsample_per_group_spatial(self, source_type: str) -> None:
        """Checks that raster and point cloud stats() include observed groups that receive a zero-size sample."""

        # Include a single-location group whose quarter sample rounds down to zero
        values = np.arange(12, dtype=float).reshape(3, 4)
        groups = np.array([0] * 7 + [1] * 4 + [2]).reshape(values.shape)
        if source_type in {"raster", "xarray"}:
            source = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 3), 32631)
            if source_type == "xarray":
                source = source.to_xarray().rst
        else:
            source = gu.PointCloud.from_xyz(np.arange(12), np.zeros(12), values.ravel(), crs=32631)
            groups = groups.ravel()
            if source_type == "geopandas":
                source = source.ds.pc

        # Leave observed=True so an unsampled group must be distinguished from an absent category
        table = source.stats(
            "mean",
            by={"category": groups},
            categories={"category": [0, 1, 2, 3]},
            subsample=0.25,
            subsample_per_group=True,
            random_state=42,
        )

        # Check that all three original groups are present, with no estimate for the unsampled one
        assert table.index.tolist() == [0, 1, 2]
        assert np.array_equal(table.xs("count", level="statistic", axis=1).iloc[:, 0], [1, 1, 0])
        assert np.isnan(table.xs("mean", level="statistic", axis=1).iloc[2, 0])

    @pytest.mark.parametrize("subsampling_strategy", ["topk", "sequential"])
    def test_stats__separate_sampling_and_reduction(self, subsampling_strategy: str) -> None:
        """Checks that the sampling option stays separate from the grouped calculation strategy."""

        # Sample ten values while requesting masks for both complete groups
        values = np.arange(100, dtype=float)
        result, masks = gu.stats.stats(
            values,
            by={"zone": values % 2 == 0},
            statistics="mean",
            subsample=10,
            random_state=42,
            strategy="sparse",
            subsampling_strategy=subsampling_strategy,
            return_masks=True,
        )

        # Check the sampled count, full masks, and recorded sampling option
        assert result[("value", "count")].sum() == 10
        assert sum(np.count_nonzero(mask) for mask in masks.values()) == 100
        assert result.attrs["grouped_stats"]["subsampling_strategy"] == subsampling_strategy

    def test_stats__vector_union_and_feature_ids(self) -> None:
        """Checks that a vector alone creates inside/outside groups while feature IDs create separate zones."""

        # Create two one-cell vector features separated by uncovered raster cells
        raster = gu.Raster.from_array(np.arange(1, 9, dtype=float).reshape(2, 4), Affine(1, 0, 0, 0, -1, 2), 32631)
        zones = gu.Vector(
            gpd.GeoDataFrame({"id": ["first", "second"]}, geometry=[box(0, 1, 1, 2), box(3, 0, 4, 1)], crs=32631)
        )

        # Group once by all vector coverage and once by each feature name
        union = raster.stats("mean", by={"inside": zones})
        features = raster.stats("mean", by={"zone": (zones, "id")})

        # Check that coverage includes outside cells while named features include only their own cells
        assert union[("band_1", "count")].tolist() == [6, 2]
        assert features[("band_1", "count")].tolist() == [1, 1]
        assert features[("band_1", "mean")].tolist() == [1, 8]

    @pytest.mark.parametrize("source_type", ["pointcloud", "dataframe"])
    def test_stats__geometry_z_is_unchanged_by_masks(self, source_type: str) -> None:
        """Checks that masks of geometry Z values add a boolean column without discarding original elevations."""

        # Existing attribute names must remain unchanged when the default new mask-column name is already taken
        points: Any = gu.PointCloud.from_xyz(np.arange(3), np.zeros(3), np.arange(3) + 100, crs=32631, use_z=True)
        points.ds["group_mask"] = [10, 11, 12]
        original = points.ds.copy()
        if source_type == "dataframe":
            points = points.ds.pc

        # Make the separate boolean mask column active without changing any geometry Z elevation
        _, masks = points.stats("mean", by={"zone": np.array([True, False, True])}, return_masks=True)
        result = masks[True]
        output = result.ds if source_type == "pointcloud" else result
        interface = result if source_type == "pointcloud" else result.pc
        assert interface.data_column == "_group_mask"
        assert interface.is_mask
        pd.testing.assert_series_equal(output.geometry, original.geometry)
        pd.testing.assert_series_equal(output.group_mask, original.group_mask)
        assert np.array_equal(output._group_mask, [True, False, True])

    def test_plot_grouped_stats__one_and_two_dimensions(self) -> None:
        """Checks that plotting one or two grouping variables creates the expected panels."""

        # Load the optional plotting package only for this plotting test
        pytest.importorskip("matplotlib")
        import matplotlib.pyplot as plt

        # Plot one grouping variable and check its count and statistic panels
        one_dimensional = gu.stats.stats(
            np.arange(6, dtype=float),
            by={"x": np.arange(6, dtype=float)},
            bins={"x": [0, 3, 6]},
            statistics="mean",
        )
        axes_1d = gu.stats.plot_grouped_stats(one_dimensional, statistic="mean")
        assert set(axes_1d) == {"count", "statistic"}

        # Plot two grouping variables and check row counts, column counts, statistic, and color scale
        two_dimensional = gu.stats.stats(
            np.arange(6, dtype=float),
            by={"x": np.array([0, 0, 1, 1, 2, 2]), "surface": np.array(["a", "b", "a", "b", "a", "b"])},
            bins={"x": [0, 1, 2, 3]},
            categories={"surface": ["a", "b"]},
            statistics="mean",
        )
        axes_2d = gu.stats.plot_grouped_stats(two_dimensional, statistic="mean")
        assert set(axes_2d) == {"count_x", "count_y", "statistic", "colorbar"}
        plt.close("all")

    @pytest.mark.parametrize("kind", ["integer", "boolean", "string"])
    def test_stats__masked_values_and_categories(self, kind: str) -> None:
        """Checks that masked values and masked category labels are excluded independently."""

        # Mask one selected value and a different group label for three category types
        values = np.ma.array([1, 2, 3, 4, 5, 6], mask=[False, True, False, False, False, False])
        group_values = {
            "integer": [0, 0, 0, 1, 1, 1],
            "boolean": [False, False, False, True, True, True],
            "string": ["a", "a", "a", "b", "b", "b"],
        }
        groups = np.ma.array(group_values[kind], mask=[False, False, True, False, False, False])
        categories = {"integer": {"group": [0, 1]}, "boolean": None, "string": {"group": ["a", "b", "N/A"]}}

        # Calculate group means and request masks for the remaining category locations
        table, masks = gu.stats.stats(
            values,
            by={"group": groups},
            categories=categories[kind],
            statistics="mean",
            return_masks=True,
        )

        # Check value counts, means, and the sizes of complete group masks separately
        assert table[("value", "count")].tolist() == [1, 3]
        assert table[("value", "mean")].tolist() == [1, 5]
        assert [int(np.count_nonzero(masks[key])) for key in masks] == [2, 3]

    def test_raster_stats__masked_integer_data_and_boolean_mask(self) -> None:
        """Checks that raster value masks and boolean user masks affect counts and group masks separately."""

        # Mask one integer value and exclude two different cells through a boolean Raster mask
        data = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[False, True, False], [False, False, False]])
        raster = gu.Raster.from_array(data, Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        mask = raster.from_array(
            np.ma.array([[True, True, True], [True, False, True]], mask=[[False, False, True], [False, False, False]]),
            raster.transform,
            raster.crs,
        )

        # Calculate one group mean and request all locations allowed by the user mask
        table, masks = raster.stats(
            "mean",
            by={"group": np.zeros(data.shape, dtype=int)},
            categories={"group": [0]},
            mask=mask,
            return_masks=True,
        )

        # Check three available values in four group locations
        assert table[("band_1", "count")].tolist() == [3]
        assert table[("band_1", "mean")].tolist() == [pytest.approx(11 / 3)]
        assert int(np.count_nonzero(masks[0].data)) == 4

    def test_stats__flox_categories(self) -> None:
        """Checks that Flox returns the same ordered category counts, means and standard deviations as GeoUtils."""

        # Create three categories in a different order from their declaration, with one selected nodata value
        pytest.importorskip("flox")
        values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
        groups = np.array([2, 2, 0, 0, 1, 1])
        options = {
            "statistics": ["mean", "std"],
            "by": {"zone": groups},
            "categories": {"zone": [2, 0, 1, 3]},
            "observed": False,
        }

        # Calculate the same complete table with the built-in and Flox reducers
        expected = gu.stats.stats(values, backend="geoutils", **options)
        result = gu.stats.stats(values, backend="flox", **options)

        # Check the declared row order, empty category and every reduced value
        pd.testing.assert_frame_equal(result, expected)
        assert result.attrs["grouped_stats"]["strategy"] == "flox"

    def test_stats__flox_bins_mask_and_subsample(self) -> None:
        """Checks that Flox matches GeoUtils bin boundaries, masking and global subsampling."""

        # Combine interval bins and categories, then exclude some locations before a reproducible global sample
        pytest.importorskip("flox")
        values = {
            "first": np.arange(24, dtype=float),
            "second": np.arange(24, dtype=float) * 2,
        }
        values["first"][5] = np.nan
        distance = np.arange(24, dtype=float) % 6
        surface = np.array(["ice", "rock"] * 12)
        options = {
            "statistics": ["mean", "sum", "totalcount", "percentagevalidpoints"],
            "by": {"distance": distance, "surface": surface},
            "bins": {"distance": [0, 2, 4, 6]},
            "categories": {"surface": ["rock", "ice", "water"]},
            "mask": np.arange(24) % 5 != 0,
            "subsample": 9,
            "random_state": 42,
            "observed": False,
        }

        # Compare every sampled value and empty declared group with the built-in reducer
        expected = gu.stats.stats(values, backend="geoutils", **options)
        result = gu.stats.stats(values, backend="flox", **options)
        pd.testing.assert_frame_equal(result, expected)

    def test_stats__error_flox_options(self) -> None:
        """Checks that Flox rejects options and statistics that its grouped path cannot reproduce."""

        # Use one ordinary category input so each call reaches Flox-specific validation
        pytest.importorskip("flox")
        values = np.arange(6, dtype=float)
        grouping = {"by": {"zone": np.arange(6) % 2}, "categories": {"zone": [0, 1]}}

        # Reject global statistics, group masks, sampling within groups, multiprocessing and GeoUtils strategies
        with pytest.raises(ValueError, match="requires grouped statistics"):
            gu.stats.stats(values, "mean", backend="flox")
        for options in (
            {"return_masks": True},
            {"subsample_per_group": True},
            {"mp_config": MultiprocConfig(chunks=2)},
            {"strategy": "dense"},
        ):
            with pytest.raises(ValueError, match="Flox backend requires"):
                gu.stats.stats(values, "mean", backend="flox", **grouping, **options)
        with pytest.raises(ValueError, match="does not support"):
            gu.stats.stats(values, "nmad", backend="flox", **grouping)

    def test_raster_stats__flox_loading_warning(self) -> None:
        """Checks that a Raster input warns that the Flox backend loads its values."""

        # Create a small Raster whose cells belong to two boolean categories
        pytest.importorskip("flox")
        raster = gu.Raster.from_array(np.arange(6, dtype=float).reshape(2, 3), Affine.identity(), 32631)
        groups = np.arange(6).reshape(2, 3) % 2 == 0

        # Calculate the grouped mean and check the warning is raised before spatial values are selected
        with pytest.warns(UserWarning, match="loads Raster and PointCloud inputs"):
            result = raster.stats("mean", by={"zone": groups}, backend="flox")
        assert result[("band_1", "count")].tolist() == [3, 3]


class TestGroupedStatsChunked:
    """
    Tests grouped statistics from stats(by=) with Dask and Multiproc inputs.

    The tests compare group definitions, common support, sampling, vector zones, returned masks, and the optional Flox
    backend with eager results. Numerical reduction strategies are covered in test_reduction.py.
    """

    @pytest.mark.parametrize("strategy", ["dense", "sparse", "groupwise"])
    def test_stats__empty_selection(self, strategy: str) -> None:
        """Checks that fully masked chunks return an empty table and mask mapping."""

        # Mask every location while requesting both boolean groups in the result
        da = pytest.importorskip("dask.array")
        values = da.ones((5, 6), chunks=2)
        table, masks = gu.stats.stats(
            values,
            by={"zone": np.ones((5, 6), dtype=bool)},
            mask=np.zeros((5, 6), dtype=bool),
            statistics="mean",
            strategy=strategy,
            return_masks=True,
        )
        eager_table, _ = gu.stats.stats(
            np.ones((5, 6)),
            by={"zone": np.ones((5, 6), dtype=bool)},
            mask=np.zeros((5, 6), dtype=bool),
            statistics="mean",
            strategy=strategy,
            return_masks=True,
        )

        # Check the empty table columns and mask mapping
        assert table.empty
        pd.testing.assert_frame_equal(table, eager_table, check_exact=True)
        assert list(table.columns) == [("value", "count"), ("value", "mean")]
        assert len(masks) == 0

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_stats__empty_arrays(self, backend: str) -> None:
        """Checks that empty inputs include requested groups with zero counts."""

        # Calculate statistics for one requested group from empty Eager, Dask, or Multiproc inputs
        values = np.empty(0)
        expected = gu.stats.stats(
            values,
            by={"zone": np.empty(0)},
            categories={"zone": [0]},
            statistics=["mean", "totalcount", "validcount"],
            observed=False,
        )
        config = MultiprocConfig(chunks=4) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            values = da.from_array(values, chunks=4)
        result = gu.stats.stats(
            values,
            by={"zone": np.empty(0)},
            categories={"zone": [0]},
            statistics=["mean", "totalcount", "validcount"],
            observed=False,
            mp_config=config,
        )
        pd.testing.assert_frame_equal(result, expected, check_exact=True)

        # Check zero counts and a NaN mean for the empty group
        assert result.loc[0, ("value", "count")] == 0
        assert result.loc[0, ("value", "totalcount")] == 0
        assert result.loc[0, ("value", "validcount")] == 0
        assert np.isnan(result.loc[0, ("value", "mean")])

    def test_stats__dask_lazy_masks_match_eager(self) -> None:
        """Checks that the Dask backend matches eager calculation and creates group masks only when requested."""

        # Calculate an eager reference from three numeric intervals
        da = pytest.importorskip("dask.array")
        values = np.arange(12, dtype=float).reshape(3, 4)
        grouper = np.arange(12, dtype=float).reshape(3, 4)
        expected = gu.stats.stats(values, by={"x": grouper}, bins={"x": [0, 4, 8, 12]}, statistics="mean")

        # Repeat with different Dask chunks for values and group labels
        table, masks = gu.stats.stats(
            da.from_array(values, chunks=(2, 2)),
            by={"x": da.from_array(grouper, chunks=(1, 4))},
            bins={"x": [0, 4, 8, 12]},
            statistics="mean",
            return_masks=True,
        )

        # Check the complete table and load only the first returned mask
        pd.testing.assert_frame_equal(table, expected)
        first_mask = masks[next(iter(masks))]
        assert isinstance(first_mask, da.Array)
        assert int(first_mask.sum().compute()) == 4

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("kind", ["boolean", "categorical"])
    @pytest.mark.parametrize("bins", [[-0.5, 1.5], 1])
    def test_stats__explicit_bins_override_category_dtype(self, backend: str, kind: str, bins: Any) -> None:
        """Checks that explicit numeric bins take precedence over boolean and Pandas categorical group types."""

        # Put both distinct labels in one numeric interval, with an unused Pandas category available for inference
        groups: Any = np.array([False, True, False, True])
        if kind == "categorical":
            groups = pd.Series(pd.Categorical([0, 1, 0, 1], categories=[0, 1, 2]))
        expected = gu.stats.stats(
            np.arange(4, dtype=float), by={"group": groups}, bins={"group": bins}, statistics="mean"
        )
        config = MultiprocConfig(chunks=3) if backend == "multiproc" else None
        if backend == "dask":
            import_optional("dask")
            import dask.array as da
            import dask.dataframe as dd

            groups = dd.from_pandas(groups, npartitions=2) if kind == "categorical" else da.from_array(groups, chunks=3)

        # Check that explicit bins produce one interval for every input type and existing category definition
        table = gu.stats.stats(
            np.arange(4, dtype=float), by={"group": groups}, bins={"group": bins}, statistics="mean", mp_config=config
        )
        pd.testing.assert_frame_equal(table, expected, check_exact=True)
        assert isinstance(table.index, pd.IntervalIndex)
        assert table[("value", "count")].tolist() == [4]
        assert table[("value", "mean")].tolist() == [1.5]

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_stats__explicit_categories_override_existing_order(self, backend: str) -> None:
        """Checks that explicit categories replace the existing order and preserve requested absent categories."""

        # Exclude one existing label from the requested categories and put a different absent label first
        groups: Any = pd.Series(pd.Categorical(["west", "east", "west"], categories=["west", "east", "north"]))
        categories = ["south", "west"]
        expected = gu.stats.stats(
            np.array([1, 2, 3]),
            by={"zone": groups},
            categories={"zone": categories},
            statistics="sum",
            observed=False,
        )
        config = MultiprocConfig(chunks=2) if backend == "multiproc" else None
        if backend == "dask":
            import_optional("dask")
            import dask.dataframe as dd

            groups = dd.from_pandas(groups, npartitions=2)

        # Pass the requested categories as an iterator and check that it is consumed only once
        table = gu.stats.stats(
            np.array([1, 2, 3]),
            by={"zone": groups},
            categories={"zone": iter(categories)},
            statistics="sum",
            observed=False,
            mp_config=config,
        )
        pd.testing.assert_frame_equal(table, expected, check_exact=True)
        assert table.index.tolist() == ["south", "west"]
        assert table[("value", "count")].tolist() == [0, 2]
        assert table[("value", "sum")].iloc[1] == 4

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("masked", [False, True])
    def test_stats__masked_integer_labels_are_exact(self, backend: str, masked: bool) -> None:
        """Checks that masked integer categories above float precision stay distinct and exclude masked labels."""

        # Adjacent integers above 2**53 collapse if a nodata label first promotes the array to float
        labels = [2**53, 2**53 + 1]
        groups: Any = np.ma.array([labels[0], labels[1], labels[1]], mask=[False, False, masked])
        expected, expected_masks = gu.stats.stats(
            np.array([1, 2, 4]),
            by={"zone": groups},
            categories={"zone": labels},
            statistics="sum",
            return_masks=True,
        )
        config = MultiprocConfig(chunks=2) if backend == "multiproc" else None
        if backend == "dask":
            import_optional("dask")
            import dask.array as da

            groups = da.from_array(groups, chunks=2)

        # Read both the independent category counts and all locations in each group
        table, masks = gu.stats.stats(
            np.array([1, 2, 4]),
            by={"zone": groups},
            categories={"zone": labels},
            statistics="sum",
            return_masks=True,
            mp_config=config,
        )
        pd.testing.assert_frame_equal(table, expected, check_exact=True)
        for label in labels:
            assert np.array_equal(np.asarray(masks[label]), np.asarray(expected_masks[label]))
        assert table.index.tolist() == labels
        assert table[("value", "count")].tolist() == [1, 1 if masked else 2]
        assert table[("value", "sum")].tolist() == [1, 2 if masked else 6]
        assert np.array_equal(np.asarray(masks[labels[1]]), [False, True, not masked])

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_stats__tuple_categories_are_single_labels(self, backend: str) -> None:
        """Checks that hashable tuple categories form one categorical level instead of a Pandas MultiIndex."""

        # Assign tuples into a one-dimensional object array so each pair is treated as one input label
        groups: Any = np.empty(3, dtype=object)
        groups[:] = [(1, 2), (3, 4), (1, 2)]
        labels = [(3, 4), (1, 2), (5, 6)]
        expected, expected_masks = gu.stats.stats(
            np.array([1, 2, 3]),
            by={"zone": groups},
            categories={"zone": labels},
            statistics="sum",
            observed=False,
            return_masks=True,
        )
        config = MultiprocConfig(chunks=2) if backend == "multiproc" else None
        if backend == "dask":
            import_optional("dask")
            import dask.array as da

            groups = da.from_array(groups, chunks=2)

        # Request one absent tuple and check that the specified order also works for mask lookup
        table, masks = gu.stats.stats(
            np.array([1, 2, 3]),
            by={"zone": groups},
            categories={"zone": labels},
            statistics="sum",
            observed=False,
            return_masks=True,
            mp_config=config,
        )
        pd.testing.assert_frame_equal(table, expected, check_exact=True)
        for label in masks:
            assert np.array_equal(np.asarray(masks[label]), np.asarray(expected_masks[label]))
        assert isinstance(table.index, pd.CategoricalIndex)
        assert table.index.tolist() == labels
        assert table[("value", "count")].tolist() == [1, 2, 0]
        assert np.array_equal(np.asarray(masks[(1, 2)]), [True, False, True])

    @pytest.mark.parametrize("source_type", ["pointcloud", "dataframe", "dask"])
    def test_stats__point_column_existing_categories(self, source_type: str) -> None:
        """Checks that point columns return categories in their existing order, including absent categories."""

        # Select an ordered category column with an absent category preceding the observed labels
        dataframe = gpd.GeoDataFrame(
            {
                "height": [1, 2, 3],
                "zone": pd.Categorical(["west", "east", "west"], categories=["north", "east", "west"]),
            },
            geometry=gpd.points_from_xy(np.arange(3), np.zeros(3)),
            crs=32631,
        )
        points: Any = gu.PointCloud(dataframe, data_column="height")
        expected = points.stats("mean", by={"zone": "zone"}, observed=False)
        if source_type == "dataframe":
            points = points.ds.pc
        elif source_type == "dask":
            dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor,
            )

            _register_dask_pointcloud_accessor()
            points = dgpd.from_geopandas(points.ds, npartitions=2, sort=False).pc
            points.data_column = "height"

        # Infer the same definition from the named column as from its original Pandas categorical values
        table = points.stats("mean", by={"zone": "zone"}, observed=False)
        pd.testing.assert_frame_equal(table, expected, check_exact=True)
        if source_type == "dask":
            assert not points.is_loaded
        assert table.index.tolist() == ["north", "east", "west"]
        assert table[("height", "count")].tolist() == [0, 1, 2]
        np.testing.assert_allclose(table[("height", "mean")], [np.nan, 2, 2], equal_nan=True)

    def test_stats__known_dask_categories(self) -> None:
        """Checks that known Dask categories return their specified order and absent groups like Pandas categories."""

        # Declare an unused category so discovering only observed labels would lose part of the requested groups
        import_optional("dask")
        import dask.dataframe as dd

        values = np.array([1.0, 3.0, 5.0, 7.0])
        categories = ["low", "high", "unused"]
        groups = pd.Series(pd.Categorical(["low", "low", "high", "high"], categories=categories, ordered=True))
        lazy_groups = dd.from_pandas(groups, npartitions=2)

        # Infer the same category order from the Pandas and Dask grouping variables
        expected = gu.stats.stats(values, "mean", by={"zone": groups}, observed=False)
        result = gu.stats.stats(values, "mean", by={"zone": lazy_groups}, observed=False)

        # Check that the unused category has zero count and a NaN estimate in the original category order
        pd.testing.assert_frame_equal(result, expected)
        assert result.index.tolist() == categories
        expected_values = np.array([[2, 2], [2, 6], [0, np.nan]], dtype=float)
        np.testing.assert_allclose(result["value"], expected_values, equal_nan=True)

    def test_stats__external_values_on_common_support(self, tmp_path: Path) -> None:
        """Checks that Dask point inputs use the same common support and return the same result as eager points."""

        # Place points at known raster cells and give two polygons different numeric values
        raster = gu.Raster.from_array(np.arange(16, dtype=float).reshape(4, 4), Affine(1, 0, 0, 0, -1, 4), 32631)
        x, y = raster.ij2xy([0, 1, 2, 3], [0, 0, 3, 3])
        frame = gpd.GeoDataFrame(
            {"height": [100.0, np.nan, 102.0, 103.0], "zone": ["west", "west", "east", "east"]},
            geometry=gpd.points_from_xy(x, y),
            crs=32631,
        )
        points = gu.PointCloud(frame, data_column="height")
        features = gu.Vector(
            gpd.GeoDataFrame(
                {"weight": [2.0, 4.0]},
                geometry=[box(0, 0, 2, 4), box(2, 0, 4, 4)],
                crs=32631,
            )
        )
        expected_table, expected_masks = raster.stats(
            by={"zone": (points, "zone")},
            categories={"zone": ["west", "east"]},
            values={"raster": raster, "points": points, "weight": (features, "weight")},
            statistics="mean",
            interpolation="nearest",
            return_masks=True,
        )

        # Write a point file to disk with height and zone columns, then reopen it as Dask partitions
        pytest.importorskip("dask_geopandas")
        filename = tmp_path / "points.gpkg"
        points.ds.to_file(filename)
        points = gu.open_pointcloud(str(filename), chunks=2, data_column="height").pc
        assert not points.is_loaded

        # Let the external point data define the common support for all selected values and grouping variables
        table, masks = raster.stats(
            by={"zone": (points, "zone")},
            categories={"zone": ["west", "east"]},
            values={"raster": raster, "points": points, "weight": (features, "weight")},
            statistics="mean",
            interpolation="nearest",
            return_masks=True,
        )
        pd.testing.assert_frame_equal(table, expected_table, check_exact=True)

        # Check each mean and count separately, then check the complete western point mask
        np.testing.assert_allclose(table[("raster", "mean")], [2.0, 13.0])
        np.testing.assert_allclose(table[("points", "mean")], [100.0, 102.5])
        np.testing.assert_allclose(table[("weight", "mean")], [2.0, 4.0])
        assert table[("points", "count")].tolist() == [1, 2]
        mask = masks["west"]
        mask_values = mask.pc.data
        dd = pytest.importorskip("dask.dataframe")
        assert isinstance(mask_values, dd.Series)
        assert not mask.pc.is_loaded
        mask_values = mask_values.compute()
        assert np.array_equal(mask_values, [True, True, False, False])
        assert np.array_equal(mask_values, expected_masks["west"].data)
        assert not points.is_loaded and not mask.pc.is_loaded

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_stats__vector_numeric_bins_and_values(self, backend: str) -> None:
        """Checks that a numeric vector column can define bins and also appear as a selected value."""

        # Give two separated polygons numeric values and leave one raster column uncovered
        raster = gu.Raster.from_array(np.arange(12, dtype=float).reshape(3, 4), Affine(1, 0, 0, 0, -1, 3), 32631)
        features = gu.Vector(
            gpd.GeoDataFrame(
                {"slope": [5.0, 15.0]},
                geometry=[box(0, 0, 1, 3), box(2, 0, 4, 3)],
                crs=32631,
            )
        )
        expected = raster.stats(
            by={"slope": (features, "slope")},
            bins={"slope": [0, 10, 20]},
            values={"raster": 1, "slope": (features, "slope")},
            statistics="mean",
        )
        if backend == "dask":
            raster = raster.to_xarray().chunk({"x": 2, "y": 2}).rst

        # Use one Multiproc tile size to read vector values, assign group IDs, and calculate statistics
        config = MultiprocConfig(chunks=(2, 3)) if backend == "multiproc" else None

        # Use the vector column as numeric bins and as a selected output value
        table = raster.stats(
            by={"slope": (features, "slope")},
            bins={"slope": [0, 10, 20]},
            values={"raster": 1, "slope": (features, "slope")},
            statistics="mean",
            mp_config=config,
        )
        pd.testing.assert_frame_equal(table, expected, check_exact=True)

        # Check group counts and both means only where a polygon covers the raster
        assert table[("raster", "count")].tolist() == [3, 6]
        np.testing.assert_allclose(table[("raster", "mean")], [4.0, 6.5])
        np.testing.assert_allclose(table[("slope", "mean")], [5.0, 15.0])

    def test_stats__vector_projection_shared_by_point_partitions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that vectors are reprojected once and later features provide values where polygons overlap."""

        # Place points inside each polygon, inside their overlap, and outside both, away from polygon boundaries
        features = gpd.GeoDataFrame(
            {"weight": [10.0, 20.0]}, geometry=[box(-1, -1, 2, 1), box(1, -1, 3, 1)], crs=4326, index=[9, 9]
        )
        dataframe = gpd.GeoDataFrame(
            {"height": np.ones(4)},
            geometry=gpd.points_from_xy([-0.5, 1.5, 2.5, 4.0], np.zeros(4)),
            crs=4326,
            index=[7, 7, 2, 2],
        ).to_crs(3857)

        expected = dataframe.pc.stats(
            "mean",
            values={"weight": (features, "weight")},
            by={"point": np.arange(4)},
            categories={"point": range(4)},
            observed=False,
        )

        # Divide the projected points into Dask partitions without changing duplicate labels or row order
        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from geoutils.pointcloud.pd_accessor import (
            _register_dask_pointcloud_accessor,
        )

        _register_dask_pointcloud_accessor()
        points = dgpd.from_geopandas(dataframe, npartitions=3, sort=False).pc
        assert not points.is_loaded

        # Count feature projections so each partition cannot repeat the same coordinate conversion
        projections = []
        to_crs = gpd.GeoDataFrame.to_crs

        def record_projection(dataframe: gpd.GeoDataFrame, *args: Any, **kwargs: Any) -> gpd.GeoDataFrame:
            """Record each feature projection while applying the usual coordinate transformation."""

            projections.append(dataframe.crs)
            return to_crs(dataframe, *args, **kwargs)

        monkeypatch.setattr(gpd.GeoDataFrame, "to_crs", record_projection)

        # Use one group per input row to check point order independently of repeated dataframe labels
        table = points.stats(
            "mean",
            values={"weight": (features, "weight")},
            by={"point": np.arange(4)},
            categories={"point": range(4)},
            observed=False,
        )
        pd.testing.assert_frame_equal(table, expected, check_exact=True)

        # The later feature supplies the overlap value, while the uncovered point has no finite observation
        assert projections == [features.crs]
        np.testing.assert_allclose(table[("weight", "mean")], [10.0, 20.0, 20.0, np.nan], equal_nan=True)
        assert np.array_equal(table[("weight", "count")], [1, 1, 1, 0])
        assert not points.is_loaded

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("sampling_strategy", ["topk", "sequential"])
    @pytest.mark.parametrize("subsample", [0.25, 3, 1])
    def test_stats__subsample_per_group(self, backend: str, sampling_strategy: str, subsample: int | float) -> None:
        """Checks that each combined group receives its own sample size while masks include every eligible location."""

        # 1/ Prepare unequal groups, one absent category/bin combination, and values with different finite counts
        groups = np.repeat(np.arange(5), [20, 14, 8, 5, 1]).reshape(6, 8)
        position = np.arange(groups.size, dtype=float).reshape(groups.shape)
        missing = position.copy()
        missing[groups == 1] = np.nan
        values = {"position": position, "doubled": 2 * position + 1, "missing": missing}
        by = {"category": groups // 2, "height": groups % 2}
        keep = np.ones(groups.shape, dtype=bool)
        keep.flat[[0, 22]] = False
        expected_table, expected_masks = gu.stats.stats(
            values,
            ["mean", "totalcount"],
            by=by,
            categories={"category": [0, 1, 2]},
            bins={"height": [-0.5, 0.5, 1.5]},
            mask=keep,
            subsample=subsample,
            subsample_per_group=True,
            subsampling_strategy=sampling_strategy,
            random_state=42,
            observed=False,
            return_masks=True,
        )

        # Cross group boundaries with uneven chunks, including differently chunked Dask grouping variables
        config = None
        if backend == "dask":
            import_optional("dask")
            import dask.array as da

            values = {name: da.from_array(array, chunks=(2, 3)) for name, array in values.items()}
            by = {name: da.from_array(array, chunks=(3, 2)) for name, array in by.items()}
        elif backend == "multiproc":
            config = MultiprocConfig(chunks=(2, 3))

        # 2/ Apply the sample size to each category and bin combination and request its complete group mask
        table, masks = gu.stats.stats(
            values,
            ["mean", "totalcount"],
            by=by,
            categories={"category": [0, 1, 2]},
            bins={"height": [-0.5, 0.5, 1.5]},
            mask=keep,
            subsample=subsample,
            subsample_per_group=True,
            subsampling_strategy=sampling_strategy,
            random_state=42,
            observed=False,
            return_masks=True,
            mp_config=config,
        )
        if sampling_strategy == "topk" or subsample == 1:
            pd.testing.assert_frame_equal(table, expected_table, check_exact=True)
        for key in masks:
            assert np.array_equal(np.asarray(masks[key]), np.asarray(expected_masks[key]))

        # 3/ Check sample sizes independently from group sizes, including the one-location and absent groups
        for number, key in enumerate(table.index):
            members = (groups == number) & keep
            size = np.count_nonzero(members)
            expected_count = int(size * subsample) if subsample <= 1 else min(int(subsample), size)
            assert table.loc[key, ("position", "count")] == expected_count
            assert table.loc[key, ("doubled", "count")] == expected_count
            assert table.loc[key, ("missing", "totalcount")] == expected_count
            assert np.array_equal(np.asarray(masks[key]), members)

            # Selected nodata values do not change the sample size or locations used for other columns
            assert table.loc[key, ("missing", "count")] == (0 if number == 1 else expected_count)
            if expected_count:
                expected_mean = 2 * table.loc[key, ("position", "mean")] + 1
                assert table.loc[key, ("doubled", "mean")] == pytest.approx(expected_mean)
            else:
                assert np.isnan(table.loc[key, ("position", "mean")])
        assert table.attrs["grouped_stats"]["subsample_per_group"] is True

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("rows", [1, 2])
    def test_stats__vector_integer_labels_outside_coverage(self, backend: str, rows: int) -> None:
        """Checks that adjacent large vector labels remain distinct when some cells fall outside every feature."""

        # Use integer labels that cannot both be represented as floats and leave the final raster column uncovered
        labels = [2**53, 2**53 + 1]
        zones = gpd.GeoDataFrame({"label": labels}, geometry=[box(0, 0, 1, 2), box(1, 0, 2, 2)], crs=32631)
        values = np.array([[1, 2, 4], [5, 6, 8]])[:rows]
        source: Any = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 2), 32631)
        expected_table, expected_masks = source.stats("sum", by={"zone": (zones, "label")}, return_masks=True)
        config = MultiprocConfig(chunks=(1, 2)) if backend == "multiproc" else None
        if backend == "dask":
            import_optional("dask")
            import dask.array as da

            source = source.to_xarray().chunk({"y": 1, "x": 2}).rst

        # Infer ordered feature categories and check that uncovered cells contribute to neither group
        table, masks = source.stats("sum", by={"zone": (zones, "label")}, return_masks=True, mp_config=config)
        pd.testing.assert_frame_equal(table, expected_table, check_exact=True)
        assert table.index.tolist() == labels
        assert table[("band_1", "count")].tolist() == [rows, rows]
        assert table[("band_1", "sum")].tolist() == values[:, :2].sum(axis=0).tolist()
        mask = masks[labels[1]]
        mask_values = mask.rst.data if backend == "dask" else mask.data
        expected_mask = expected_masks[labels[1]].data
        if backend == "dask":
            assert isinstance(mask_values, da.Array)
            mask_values = mask_values.compute()
        assert np.array_equal(np.asarray(mask_values), np.asarray(expected_mask))
        assert np.array_equal(np.asarray(mask_values).reshape(-1), np.tile([False, True, False], rows))

    @pytest.mark.parametrize("grouping", ["categories", "bins"])
    def test_stats__vector_groups_file_points_not_loaded(self, grouping: str, tmp_path: Path) -> None:
        """Checks that vector categories and bins match eager while a point file remains unloaded."""

        # Write a point file to disk with four values and the final point outside both vector features
        dataframe = gpd.GeoDataFrame(
            {"height": [1, 2, 3, 4]}, geometry=gpd.points_from_xy(np.arange(4) + 0.5, np.full(4, 0.5)), crs=32631
        )
        path = tmp_path / "vector_group_points.gpkg"
        dataframe.to_file(path)
        source = gu.PointCloud(path, data_column="height")
        zones = gpd.GeoDataFrame(
            {"label": ["west", "east"], "number": [10, 20]},
            geometry=[box(0, 0, 2, 1), box(2, 0, 3, 1)],
            crs=32631,
        )
        column = "label" if grouping == "categories" else "number"
        bins = None if grouping == "categories" else {"zone": [5, 15, 25]}
        expected = gu.PointCloud(dataframe, data_column="height").stats("sum", by={"zone": (zones, column)}, bins=bins)

        # Read coordinates and values in two-point blocks when assigning vector groups
        table = source.stats("sum", by={"zone": (zones, column)}, bins=bins, mp_config=MultiprocConfig(chunks=2))
        pd.testing.assert_frame_equal(table, expected, check_exact=True)

        # The first feature contains two points, the second contains one, and the source stays unloaded
        assert not source.is_loaded
        assert table[("height", "count")].tolist() == [2, 1]
        assert table[("height", "sum")].tolist() == [3, 3]

    def test_stats__vector_coverage_includes_point_boundaries(self, tmp_path: Path) -> None:
        """Checks that vector coverage groups include boundary points for eager and unloaded point values."""

        # Write a point file to disk with one point on the feature edge, one inside, and one outside
        dataframe = gpd.GeoDataFrame(
            {"height": [1, 2, 4]}, geometry=gpd.points_from_xy([0, 0.5, 2], [0.5, 0.5, 0.5]), crs=32631
        )
        path = tmp_path / "coverage_points.gpkg"
        dataframe.to_file(path)
        source = gu.PointCloud(path, data_column="height")
        outline = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=32631))
        expected = gu.PointCloud(dataframe, data_column="height").stats("sum", by={"covered": outline})

        # Group points by vector coverage and include the point exactly on the polygon boundary
        table = source.stats("sum", by={"covered": outline}, mp_config=MultiprocConfig(chunks=2))
        pd.testing.assert_frame_equal(table, expected, check_exact=True)
        assert table.index.tolist() == [False, True]
        assert table[("height", "count")].tolist() == [1, 2]
        assert table[("height", "sum")].tolist() == [4, 3]
        assert not source.is_loaded

    @pytest.mark.parametrize("source_type", ["raster", "xarray", "dask", "multiproc", "pointcloud", "geopandas"])
    @pytest.mark.parametrize("vector_object", [False, True])
    def test_stats__feature_zones(self, source_type: str, vector_object: bool) -> None:
        """Checks that vector feature names define zones with separate nodata counts and complete masks."""

        # Build repeated named zones, a gap between features, and one zone outside the data
        elevation = np.arange(1, 17, dtype=float).reshape(4, 4)
        other = elevation + 100
        elevation[0, 0], other[2, 3] = np.nan, np.nan
        zones = gpd.GeoDataFrame(
            {"id": ["west", "east", "west", "empty"]},
            geometry=[box(0, 0, 1, 2), box(3, 0, 4, 4), box(0, 2, 1, 4), box(10, 10, 11, 11)],
            crs=32631,
        )
        vector = gu.Vector(zones) if vector_object else zones

        # Store the same two value arrays as raster bands, Xarray data, or point columns
        selected_values: dict[str, int] | dict[str, str]
        if source_type in {"raster", "xarray", "dask", "multiproc"}:
            source = gu.Raster.from_array(np.stack((elevation, other)), Affine(1, 0, 0, 0, -1, 4), 32631, nodata=np.nan)
            selected_values = {"elevation": 1, "other": 2}
            expected_table, _ = source.stats(
                by={"zone": (vector, "id")},
                values=selected_values,
                statistics=("mean", "min", "max"),
                observed=False,
                return_masks=True,
            )
            if source_type in {"xarray", "dask"}:
                xarray_source = source.to_xarray()
                if source_type == "dask":
                    da = pytest.importorskip("dask.array")
                    xarray_source = xarray_source.chunk({"band": 1, "y": 3, "x": 2})
                source = xarray_source.rst
        else:
            x, y = np.meshgrid(np.arange(4) + 0.5, 3.5 - np.arange(4))
            frame = gpd.GeoDataFrame(
                {"elevation": elevation.ravel(), "other": other.ravel()},
                geometry=gpd.points_from_xy(x.ravel(), y.ravel()),
                crs=32631,
            )
            source = gu.PointCloud(frame, data_column="elevation")
            selected_values = {"elevation": "elevation", "other": "other"}
            expected_table, _ = source.stats(
                by={"zone": (vector, "id")},
                values=selected_values,
                statistics=("mean", "min", "max"),
                observed=False,
                return_masks=True,
            )
            if source_type == "geopandas":
                source = source.ds.pc

        # Use uneven Multiproc tiles so vector placement and calculation cross tile boundaries
        config = MultiprocConfig(chunks=(3, 2)) if source_type == "multiproc" else None

        # Group by the vector names, include the outside zone, and request group masks
        table, masks = source.stats(
            by={"zone": (vector, "id")},
            values=selected_values,
            statistics=("mean", "min", "max"),
            observed=False,
            return_masks=True,
            mp_config=config,
        )
        pd.testing.assert_frame_equal(table, expected_table, check_exact=True)
        assert isinstance(table.index, pd.CategoricalIndex)
        assert table.index.tolist() == ["west", "east", "empty"]

        # Check each value's nodata separately and leave the gap outside every group
        assert table[("elevation", "count")].tolist() == [3, 4, 0]
        assert table[("other", "count")].tolist() == [4, 3, 0]
        np.testing.assert_allclose(table[("elevation", "mean")], [9, 10, np.nan], equal_nan=True)
        np.testing.assert_allclose(table[("elevation", "min")], [5, 4, np.nan], equal_nan=True)
        np.testing.assert_allclose(table[("elevation", "max")], [13, 16, np.nan], equal_nan=True)

        # Check the full western zone mask, including its location with a selected nodata value
        expected = np.zeros((4, 4), dtype=bool)
        expected[:, 0] = True
        west_mask = masks["west"]
        mask_data = west_mask.pc.data if source_type == "geopandas" else west_mask.data
        assert np.array_equal(np.asarray(mask_data).reshape(4, 4), expected)
        if source_type == "dask":
            assert isinstance(source.data, da.Array)
            assert isinstance(mask_data, da.Array)

    @pytest.mark.parametrize("source_type", ["pointcloud", "dataframe", "dask"])
    @pytest.mark.parametrize("size", [1, 3])
    def test_stats__point_mask_rows_and_geometry(self, source_type: str, size: int) -> None:
        """Checks that point masks have the same 3D geometry and row order, including singleton inputs."""

        # Use duplicate row labels and unrelated attributes so assigning by index would change the result
        dataframe = gpd.GeoDataFrame(
            {"height": np.arange(size), "attribute": np.arange(size) + 10},
            geometry=gpd.points_from_xy(np.arange(size), np.zeros(size), np.arange(size) + 100),
            crs=32631,
            index=np.zeros(size, dtype=int),
        )
        dataframe.attrs["data_column"] = "height"
        if source_type == "pointcloud":
            with pytest.warns(UserWarning, match="Overriding 3D points"):
                points: Any = gu.PointCloud(dataframe, data_column="height")
        elif source_type == "dataframe":
            points = dataframe.pc
        else:
            dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor,
            )

            _register_dask_pointcloud_accessor()
            points = dgpd.from_geopandas(dataframe, npartitions=min(size, 2), sort=False).pc
            points.data_column = "height"

        # Request complete masks and replace only the active point data column with boolean mask values
        table, masks = points.stats(
            "mean", by={"zone": np.arange(size) % 2}, categories={"zone": [0, 1]}, return_masks=True
        )
        result = masks[0]
        output = result.ds if source_type == "pointcloud" else result
        if source_type == "dask":
            assert output.height.dtype == bool
            output = output.compute()

        # Group zero contains every second original row, with all coordinates and other attributes unchanged
        pd.testing.assert_series_equal(output.geometry, dataframe.geometry)
        pd.testing.assert_series_equal(output.attribute, dataframe.attribute)
        assert np.array_equal(output.height, np.arange(size) % 2 == 0)
        assert output.height.dtype == bool
        assert table[("height", "count")].sum() == size

    def test_stats__flox_matches_eager(self) -> None:
        """Checks that Flox computes a Pandas table equal to eager GeoUtils output while inputs remain Dask arrays."""

        # Create two interleaved groupers and give the selected values a different Dask chunk layout
        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")
        values = np.arange(48, dtype=float).reshape(6, 8)
        values[0, 0] = np.nan
        rows, columns = np.indices(values.shape)
        row_groups = rows % 2
        column_groups = columns % 3
        keep = columns != 1
        options = {
            "statistics": ["mean", "std", "sum", "sumofsquares", "rmse", "totalcount"],
            "by": {"row": row_groups, "column": column_groups},
            "categories": {"row": [1, 0, 2], "column": [2, 0, 1, 3]},
            "mask": keep,
            "observed": False,
        }
        expected = gu.stats.stats(values, backend="geoutils", **options)
        lazy_values = da.from_array(values, chunks=(2, 4))
        lazy_groups = {
            "row": da.from_array(row_groups, chunks=(3, 2)),
            "column": da.from_array(column_groups, chunks=(3, 2)),
        }

        # Run the Dask Flox backend, which computes only the small grouped result table
        result = gu.stats.stats(
            lazy_values,
            statistics=options["statistics"],
            by=lazy_groups,
            categories=options["categories"],
            mask=da.from_array(keep, chunks=(3, 2)),
            observed=False,
            backend="flox",
        )

        # Check input laziness, computed output type and every result against the eager calculation
        assert isinstance(lazy_values, da.Array)
        assert all(isinstance(grouper, da.Array) for grouper in lazy_groups.values())
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected)
