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
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.raster.xr_accessor import RasterAccessor


class TestChunkStrategies:
    """Compare grouping strategies on arrays independently of spatial preparation."""

    @pytest.mark.parametrize("strategy", ["auto", "dense", "sparse", "groupwise"])
    @pytest.mark.parametrize("backend", ["numpy", "dask", "multiproc"])
    @pytest.mark.parametrize("shape", [(90,), (9, 10)])
    def test_reductions(self, strategy: str, backend: str, shape: tuple[int, ...]) -> None:
        """Checks that each strategy combines groups and independent missing values correctly."""

        # Distribute groups across uneven chunks and leave one declared group unobserved
        rng = np.random.default_rng(42)
        groups = (np.arange(90) % 3).reshape(shape)
        first = rng.normal(size=shape)
        second = first * 2 + 10
        first.flat[::7] = np.nan
        second.flat[::11] = np.inf
        keep = np.arange(90).reshape(shape) % 13 != 0

        # Exercise different value and membership chunk boundaries with the same eager reference
        values = {"first": first, "second": second}
        by = {"zone": groups}
        config = None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            values = {name: da.from_array(array, chunks=4) for name, array in values.items()}
            by = {"zone": da.from_array(groups, chunks=3)}
        elif backend == "multiproc":
            config = MultiprocConfig(chunks=4)

        # Request mergeable estimators together to verify shared counts and variance combination
        statistics = ["mean", "standarddeviation", "sum", "minimum", "maximum", "rmse", "totalcount"]
        table = gu.stats.grouped_stats(
            values,
            by,
            categories={"zone": [0, 1, 2, 3]},
            statistics=statistics,
            mask=keep,
            observed=False,
            strategy=strategy,
            mp_config=config,
        )

        # Compare each populated group directly with NumPy, excluding each value's nodata separately
        for name, array in {"first": first, "second": second}.items():
            for label in range(3):
                members = array[(groups == label) & keep]
                finite = members[np.isfinite(members)]
                expected = [
                    finite.size,
                    finite.mean(),
                    finite.std(),
                    finite.sum(),
                    finite.min(),
                    finite.max(),
                    np.sqrt(np.mean(finite**2)),
                    members.size,
                ]
                np.testing.assert_allclose(table.loc[label, name], expected, rtol=1e-12, atol=1e-12)
            assert table.loc[3, (name, "count")] == 0
            assert np.isnan(table.loc[3, (name, "mean")])

    @pytest.mark.parametrize("strategy", ["dense", "sparse", "groupwise"])
    @pytest.mark.parametrize("shape,chunks", [((11, 13), (5, 5)), ((257, 509), (63, 127))])
    def test_worker_processes(self, strategy: str, shape: tuple[int, int], chunks: tuple[int, int]) -> None:
        """Checks that multiprocessing workers execute the same grouping kernels correctly."""

        # Use several uneven tiles so every group crosses worker task boundaries
        from geoutils.multiproc.cluster import MpCluster

        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        groups = np.indices(values.shape).sum(axis=0) % 3
        expected = gu.stats.grouped_stats(values, {"zone": groups}, categories={"zone": range(3)}, statistics="mean")

        # Run actual worker processes rather than only the synchronous cluster interface
        with MpCluster({"nb_workers": 2}) as cluster:
            result = gu.stats.grouped_stats(
                values,
                {"zone": groups},
                categories={"zone": range(3)},
                statistics="mean",
                strategy=strategy,
                mp_config=MultiprocConfig(chunks=chunks, cluster=cluster),
            )

        # Compare public labels, counts and estimates after gathering worker results
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_exact_statistics(self, backend: str) -> None:
        """Checks that exact medians, NMAD and custom functions receive complete groups correctly."""

        # Interleave small and large groups with missing observations across all chunks
        values = np.arange(35, dtype=float)
        values[::6] = np.nan
        groups = np.arange(35) % 4
        config = MultiprocConfig(chunks=6) if backend == "multiproc" else None
        data = values
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            data = da.from_array(values, chunks=6)

        # A custom size function detects padded full-array groups as well as dropped missing members
        table = gu.stats.grouped_stats(
            data,
            {"zone": groups},
            categories={"zone": range(4)},
            statistics=["median", "nmad", np.size],
            strategy="auto",
            mp_config=config,
        )

        # Calculate robust references on each complete group's actual members
        assert table.attrs["grouped_stats"]["strategy"] == "groupwise"
        for label in range(4):
            members = values[groups == label]
            median = np.nanmedian(members)
            expected = [
                np.isfinite(members).sum(),
                median,
                1.4826 * np.nanmedian(np.abs(members - median)),
                members.size,
            ]
            np.testing.assert_allclose(table.loc[label], expected)

    @pytest.mark.parametrize("strategy", ["dense", "sparse"])
    def test_variance_with_large_offset(self, strategy: str) -> None:
        """Checks that combining chunk variances preserves small variation around a large offset correctly."""

        # Make raw second moment subtraction lose precision while centered deviations remain measurable
        da = pytest.importorskip("dask.array")
        values = 1e8 + np.random.default_rng(42).normal(scale=0.1, size=3000)
        groups = np.arange(values.size) % 3
        result = gu.stats.grouped_stats(
            da.from_array(values, chunks=101),
            {"zone": groups},
            categories={"zone": range(3)},
            statistics="std",
            strategy=strategy,
        )

        # Compare population standard deviations independently of the block reduction algorithm
        expected = [np.std(values[groups == label]) for label in range(3)]
        np.testing.assert_allclose(result[("value", "std")], expected, rtol=1e-7)

    @pytest.mark.parametrize("strategy", ["dense", "sparse", "groupwise"])
    def test_empty_selection(self, strategy: str) -> None:
        """Checks that fully masked chunks return an empty result and empty mask mapping correctly."""

        # Exclude all membership without changing the declared categories
        da = pytest.importorskip("dask.array")
        values = da.ones((5, 6), chunks=2)
        table, masks = gu.stats.grouped_stats(
            values,
            {"zone": np.ones((5, 6), dtype=bool)},
            mask=np.zeros((5, 6), dtype=bool),
            statistics="mean",
            strategy=strategy,
            return_masks=True,
        )

        # Keep the ordinary public result structure even when no group is observed
        assert table.empty
        assert list(table.columns) == [("value", "count"), ("value", "mean")]
        assert len(masks) == 0

    def test_strategy_validation(self) -> None:
        """Checks that incompatible grouping strategies and execution backends are rejected correctly."""

        # Prepare one chunked array so invalid choices are checked on the chunked path
        da = pytest.importorskip("dask.array")
        values = da.arange(12, chunks=4)
        by = {"zone": np.arange(12) % 2 == 0}

        # Require exact group gathering for arbitrary or robust statistics
        with pytest.raises(ValueError, match="require strategy='groupwise'"):
            gu.stats.grouped_stats(values, by, statistics="median", strategy="dense")
        with pytest.raises(ValueError, match="strategy must"):
            gu.stats.grouped_stats(values, by, strategy="topk")
        with pytest.raises(ValueError, match="Dask inputs cannot"):
            gu.stats.grouped_stats(values, by, mp_config=MultiprocConfig(chunks=4))

    def test_groupwise_reads_only_intersecting_blocks(self) -> None:
        """Checks that complete groups share block reads and skip blocks outside every group correctly."""

        # Track value reads independently of the already available group membership
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        reads = []

        def read_values(block: int) -> NDArrayNum:
            """Record each value block read so unnecessary gathers remain visible."""

            # The final block has no membership and must never reach the value reader
            assert block < 2
            reads.append(block)
            return np.arange(4, dtype=float) + 4 * block

        blocks = [da.from_delayed(dask.delayed(read_values)(block), shape=(4,), dtype=float) for block in range(3)]
        values = da.concatenate(blocks)
        labels = np.array([0, 1, 0, 1, 2, 2, 2, 2, -1, -1, -1, -1])

        # Group two small zones together because their complete membership shares the first block
        with dask.config.set(scheduler="synchronous"):
            result = gu.stats.grouped_stats(
                values,
                {"zone": labels},
                categories={"zone": [0, 1, 2]},
                statistics="median",
                strategy="groupwise",
            )

        # Read each intersecting block once and preserve exact group medians
        assert sorted(reads) == [0, 1]
        np.testing.assert_allclose(result[("value", "median")], [1.0, 2.0, 5.5])

    @pytest.mark.parametrize("backend", ["numpy", "dask", "multiproc"])
    def test_empty_arrays(self, backend: str) -> None:
        """Checks that empty inputs preserve declared groups and zero counts correctly."""

        # Retain a declared group even though there are no observations to distribute across blocks
        values = np.empty(0)
        config = MultiprocConfig(chunks=4) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            values = da.from_array(values, chunks=4)
        result = gu.stats.grouped_stats(
            values,
            {"zone": np.empty(0)},
            categories={"zone": [0]},
            statistics=["mean", "totalcount", "validcount"],
            observed=False,
            mp_config=config,
        )

        # Counts have a defined zero value while an estimate without observations remains undefined
        assert result.loc[0, ("value", "count")] == 0
        assert result.loc[0, ("value", "totalcount")] == 0
        assert result.loc[0, ("value", "validcount")] == 0
        assert np.isnan(result.loc[0, ("value", "mean")])

    @pytest.mark.parametrize("subsampling_strategy", ["topk", "sequential"])
    def test_separate_subsampling_strategy(self, subsampling_strategy: str) -> None:
        """Checks that sampling controls remain independent of the chosen aggregation strategy correctly."""

        # Request sampled statistics but retain the complete two-group membership masks
        values = np.arange(100, dtype=float)
        result, masks = gu.stats.grouped_stats(
            values,
            {"zone": values % 2 == 0},
            statistics="mean",
            subsample=10,
            random_state=42,
            strategy="sparse",
            subsampling_strategy=subsampling_strategy,
            return_masks=True,
        )

        # Keep the requested sample size without changing group masks or mixing the two strategy options
        assert result[("value", "count")].sum() == 10
        assert sum(np.count_nonzero(mask) for mask in masks.values()) == 100
        assert result.attrs["grouped_stats"]["subsampling_strategy"] == subsampling_strategy


class TestInputAndChunkSizes:
    """Independent numerical references for large inputs, uneven partitions and sparse membership."""

    @pytest.mark.parametrize("shape", [(131, 197), (1025, 1537)])
    @pytest.mark.parametrize("chunks", [(37, 61), (128, 193), (2048, 2048)])
    @pytest.mark.parametrize("layout", ["local", "interleaved"])
    @pytest.mark.parametrize("strategy", ["auto", "dense", "sparse", "groupwise"])
    def test_reductions_across_sizes(
        self, shape: tuple[int, int], chunks: tuple[int, int], layout: str, strategy: str
    ) -> None:
        """Checks that all reduction modes preserve independent counts and estimates across input and chunk sizes."""

        # Use exact quarter increments and two independent nodata patterns for an unambiguous NumPy reference
        da = pytest.importorskip("dask.array")
        rows, columns = np.indices(shape)
        positions = rows * shape[1] + columns
        first = 20 + (positions % 97) * 0.25
        second = -2 * first + positions % 3
        first[positions % 17 == 0] = np.nan
        second[positions % 29 == 0] = np.inf

        # Place groups either in bounded regions or across every chunk, and exclude whole blocks as well as cells
        groups = positions % 8
        if layout == "local":
            groups = (rows * 2 // shape[0]) * 4 + columns * 4 // shape[1]
        groups[positions % 31 == 0] = -1
        keep = (rows + columns) % 19 != 0
        keep[: shape[0] // 4, : shape[1] // 4] = False

        # Deliberately mismatch value, membership and mask partitions, including one chunk larger than the input
        values = {"first": da.from_array(first, chunks=chunks), "second": da.from_array(second, chunks=chunks[::-1])}
        by = {"zone": da.from_array(groups, chunks=(chunks[0] + 3, chunks[1] + 5))}
        result = gu.stats.grouped_stats(
            values,
            by,
            categories={"zone": range(9)},
            statistics=["mean", "std", "sum", "min", "max", "rmse", "totalcount"],
            mask=da.from_array(keep, chunks=chunks[::-1]),
            strategy=strategy,
            observed=False,
        )

        # Compute each reference from the original observations, independently of chunk reducers or eager grouping
        for name, array in {"first": first, "second": second}.items():
            for label in range(8):
                members = array[(groups == label) & keep]
                finite = members[np.isfinite(members)]
                expected = [finite.mean(), finite.std(), finite.sum(), finite.min(), finite.max()]
                expected.append(np.sqrt(np.mean(finite**2)))
                assert result.loc[label, (name, "count")] == finite.size
                assert result.loc[label, (name, "totalcount")] == members.size
                np.testing.assert_allclose(result.loc[label, name].iloc[1:-1], expected, rtol=1e-12, atol=1e-12)

            # Declared empty groups survive without acquiring fabricated finite observations
            assert result.loc[8, (name, "count")] == 0
            assert result.loc[8, (name, "totalcount")] == 0
            assert np.isnan(result.loc[8, (name, "mean")])

    @pytest.mark.parametrize("declared_groups,expected_strategy", [(4096, "dense"), (4097, "sparse")])
    def test_auto_and_sparse_membership(self, declared_groups: int, expected_strategy: str) -> None:
        """Checks that automatic selection follows declared group count and sparse merges retain distant group IDs."""

        # Occupy only three widely separated category IDs and leave the first chunks entirely unclassified
        da = pytest.importorskip("dask.array")
        values = np.arange(257 * 263, dtype=float).reshape(257, 263) % 101
        labels = np.array([0, declared_groups // 2, declared_groups - 1])
        groups = labels[np.indices(values.shape).sum(axis=0) % 3]
        groups[:64, :] = -1

        # Compare explicit reducers and the threshold rule using different value and grouper partitions
        for strategy in ("auto", "dense", "sparse"):
            result = gu.stats.grouped_stats(
                da.from_array(values, chunks=(31, 47)),
                {"zone": da.from_array(groups, chunks=(73, 61))},
                categories={"zone": range(declared_groups)},
                statistics=["mean", "std"],
                strategy=strategy,
            )

            # Observed output remains small even though the declared category space determines dense allocation
            assert list(result.index) == list(labels)
            resolved = expected_strategy if strategy == "auto" else strategy
            assert result.attrs["grouped_stats"]["strategy"] == resolved
            for label in labels:
                members = values[groups == label]
                np.testing.assert_allclose(result.loc[label], [members.size, members.mean(), members.std()])

    @pytest.mark.parametrize("chunks", [(31, 47), (128, 193), (1024, 1024)])
    @pytest.mark.parametrize("strategy", ["auto", "groupwise"])
    def test_exact_statistics_across_chunks(self, chunks: tuple[int, int], strategy: str) -> None:
        """Checks that exact medians, NMAD and custom functions use complete uneven groups across chunk layouts."""

        # One large group and several small groups make medians of chunk medians an incorrect shortcut
        da = pytest.importorskip("dask.array")
        rng = np.random.default_rng(42)
        values = rng.lognormal(size=(513, 769))
        groups = np.zeros(values.shape, dtype=int)
        groups[::7, ::3] = 1
        groups[-3:, -5:] = 2
        values[::11, ::13] = np.nan

        # Include a custom function that checks full membership including missing values
        result = gu.stats.grouped_stats(
            da.from_array(values, chunks=chunks),
            {"zone": da.from_array(groups, chunks=chunks[::-1])},
            categories={"zone": range(3)},
            statistics=["median", "nmad", np.size],
            strategy=strategy,
        )

        # Derive exact robust references directly from each complete original group
        for label in range(3):
            members = values[groups == label]
            median = np.nanmedian(members)
            nmad = 1.4826 * np.nanmedian(np.abs(members - median))
            np.testing.assert_allclose(result.loc[label], [np.isfinite(members).sum(), median, nmad, members.size])


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


class TestZonalStatistics:
    """Statistics and geographic masks for bins defined by vector features."""

    @pytest.mark.parametrize("backend", ["raster", "xarray", "dask", "pointcloud", "geopandas"])
    @pytest.mark.parametrize("vector_object", [False, True])
    def test_feature_zones(self, backend: str, vector_object: bool) -> None:
        """Checks that vector feature IDs define zones with independent finite counts and complete masks."""

        # Combine two western features into one zone, leaving a gap and a zone outside the dataset
        elevation = np.arange(1, 17, dtype=float).reshape(4, 4)
        other = elevation + 100
        elevation[0, 0], other[2, 3] = np.nan, np.nan
        zones = gpd.GeoDataFrame(
            {"id": ["west", "east", "west", "empty"]},
            geometry=[box(0, 0, 1, 2), box(3, 0, 4, 4), box(0, 2, 1, 4), box(10, 10, 11, 11)],
            crs=32631,
        )
        vector = gu.Vector(zones) if vector_object else zones

        # Use the same observations as raster bands or point columns at pixel centres
        selected_values: dict[str, int] | dict[str, str]
        if backend in {"raster", "xarray", "dask"}:
            source = gu.Raster.from_array(np.stack((elevation, other)), Affine(1, 0, 0, 0, -1, 4), 32631, nodata=np.nan)
            selected_values = {"elevation": 1, "other": 2}
            if backend != "raster":
                native = source.to_xarray()
                if backend == "dask":
                    da = pytest.importorskip("dask.array")
                    native = native.chunk({"band": 1, "y": 3, "x": 2})
                source = native.rst
        else:
            x, y = np.meshgrid(np.arange(4) + 0.5, 3.5 - np.arange(4))
            frame = gpd.GeoDataFrame(
                {"elevation": elevation.ravel(), "other": other.ravel()},
                geometry=gpd.points_from_xy(x.ravel(), y.ravel()),
                crs=32631,
            )
            source = gu.PointCloud(frame, data_column="elevation")
            selected_values = {"elevation": "elevation", "other": "other"}
            if backend == "geopandas":
                source = source.ds.pc

        # Retain unobserved zones and obtain the membership mask for each feature ID
        table, masks = source.grouped_stats(
            by={"zone": (vector, "id")},
            values=selected_values,
            statistics=("mean", "min", "max"),
            observed=False,
            return_masks=True,
        )
        assert isinstance(table.index, pd.CategoricalIndex)
        assert table.index.tolist() == ["west", "east", "empty"]

        # Exclude missing values separately for each band or column, and exclude the gap between zones
        assert table[("elevation", "count")].tolist() == [3, 4, 0]
        assert table[("other", "count")].tolist() == [4, 3, 0]
        np.testing.assert_allclose(table[("elevation", "mean")], [9, 10, np.nan], equal_nan=True)
        np.testing.assert_allclose(table[("elevation", "min")], [5, 4, np.nan], equal_nan=True)
        np.testing.assert_allclose(table[("elevation", "max")], [13, 16, np.nan], equal_nan=True)

        # Keep full zone membership even where a selected value is missing
        expected = np.zeros((4, 4), dtype=bool)
        expected[:, 0] = True
        west_mask = masks["west"]
        mask_data = west_mask.pc.data if backend == "geopandas" else west_mask.data
        np.testing.assert_array_equal(np.asarray(mask_data).reshape(4, 4), expected)
        if backend == "dask":
            assert isinstance(source.data, da.Array)
            assert isinstance(mask_data, da.Array)

    def test_vector_union_and_feature_ids(self) -> None:
        """Checks that a bare vector groups inside and outside while feature IDs keep separate zones."""

        # Separate two single-cell zones by uncovered cells
        raster = gu.Raster.from_array(np.arange(1, 9, dtype=float).reshape(2, 4), Affine(1, 0, 0, 0, -1, 2), 32631)
        zones = gu.Vector(
            gpd.GeoDataFrame({"id": ["first", "second"]}, geometry=[box(0, 1, 1, 2), box(3, 0, 4, 1)], crs=32631)
        )

        # Compare the two documented ways of passing vector grouping variables
        union = raster.grouped_stats(by={"inside": zones}, statistics="mean")
        features = raster.grouped_stats(by={"zone": (zones, "id")}, statistics="mean")

        # The union includes an outside category; feature grouping excludes locations outside all zones
        assert union[("band_1", "count")].tolist() == [6, 2]
        assert features[("band_1", "count")].tolist() == [1, 1]
        assert features[("band_1", "mean")].tolist() == [1, 8]


class TestSharedSampling:
    """Prepare external values and grouping variables using the co-sampling workflow."""

    @pytest.mark.parametrize("lazy", [False, True])
    def test_external_objects(self, lazy: bool, tmp_path: Path) -> None:
        """Checks that raster, point and vector values share natural point support correctly."""

        # Place points at known raster sample locations and assign numeric polygon attributes
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
        if lazy:
            pytest.importorskip("dask_geopandas")
            filename = tmp_path / "points.gpkg"
            points.ds.to_file(filename)
            points = gu.open_pointcloud(str(filename), chunks=2, data_column="height").pc

        # An external point dataset selects point support without requiring an explicit at argument
        table, masks = raster.grouped_stats(
            by={"zone": (points, "zone")},
            categories={"zone": ["west", "east"]},
            values={"raster": raster, "points": points, "weight": (features, "weight")},
            statistics="mean",
            interpolation="nearest",
            return_masks=True,
        )

        # Keep each value's missing data independent after using the same sampling locations
        np.testing.assert_allclose(table[("raster", "mean")], [2.0, 13.0])
        np.testing.assert_allclose(table[("points", "mean")], [100.0, 102.5])
        np.testing.assert_allclose(table[("weight", "mean")], [2.0, 4.0])
        assert table[("points", "count")].tolist() == [1, 2]
        mask = masks["west"]
        mask_values = mask.pc.data if lazy else mask.data
        if lazy:
            dd = pytest.importorskip("dask.dataframe")
            assert isinstance(mask_values, dd.Series)
            mask_values = mask_values.compute()
        np.testing.assert_array_equal(mask_values, [True, True, False, False])

    @pytest.mark.parametrize("lazy", [False, True])
    def test_vector_numeric_bins_and_values(self, lazy: bool) -> None:
        """Checks that vector attributes can supply continuous bins and external raster values correctly."""

        # Assign repeated numeric attributes to separated polygons with one uncovered raster column
        raster = gu.Raster.from_array(np.arange(12, dtype=float).reshape(3, 4), Affine(1, 0, 0, 0, -1, 3), 32631)
        features = gu.Vector(
            gpd.GeoDataFrame(
                {"slope": [5.0, 15.0]},
                geometry=[box(0, 0, 1, 3), box(2, 0, 4, 3)],
                crs=32631,
            )
        )
        if lazy:
            raster = raster.to_xarray().chunk({"x": 2, "y": 2}).rst

        # Explicit bins select continuous vector values instead of treating each attribute as a zone ID
        table = raster.grouped_stats(
            by={"slope": (features, "slope")},
            bins={"slope": [0, 10, 20]},
            values={"raster": 1, "slope": (features, "slope")},
            statistics="mean",
        )

        # Exclude uncovered cells and retain the numeric attributes on their matching spatial support
        assert table[("raster", "count")].tolist() == [3, 6]
        np.testing.assert_allclose(table[("raster", "mean")], [4.0, 6.5])
        np.testing.assert_allclose(table[("slope", "mean")], [5.0, 15.0])


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
