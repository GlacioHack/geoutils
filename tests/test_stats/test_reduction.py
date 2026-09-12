"""Tests shared global and grouped statistic reductions."""

from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from affine import Affine

import geoutils as gu
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.readers import _ValueReader
from geoutils.stats.reduction import (
    _aggregate_collected,
    _collect_block,
    _finalize_blocks,
    _merge_blocks,
    _normalize_statistics,
    _reduce_block,
    _reduce_global_values,
    _reduce_values,
    _resolve_strategy,
    _Statistics,
    _statistics_dask,
)


class TestReduction:
    """
    Tests the reduction (i.e., applying statistical reducer function like mean() or std()) for eager arrays.

    The Dask/Multiproc chunked execution are tested further below in TestReductionChunked.

    Below, we check statistic names and aliases from user inputs, global and grouped calculations,
    behaviour with nodata values, and the impact of data types (integer, float) on the statistics.
    """

    def test_normalize_statistics__names_and_aliases(self) -> None:
        """Checks that strings, partial functions and callable objects get clear names and internal aliases."""

        # Build a callable example
        class Span:
            def __call__(self, values: Any) -> Any:
                return np.nanmax(values) - np.nanmin(values)

        # Test with a synonym, a partial function without its own name, and a callable object
        percentile = partial(np.nanpercentile, q=75)
        span = Span()
        statistics = _normalize_statistics(["standard deviation", percentile, span], grouped=True)

        # We verify the requests, output names, aliases used by reducers, and mandatory grouped count
        assert isinstance(statistics, _Statistics)
        assert statistics.requested == ["standard deviation", percentile, span]
        assert statistics.names == ["standard deviation", "nanpercentile", "Span"]
        assert statistics.aliases == ["std", None, None]
        assert statistics.output_names == ["count", "standard deviation", "nanpercentile", "Span"]

        # A single global statistic returns its own value without a grouped count column
        summary = _normalize_statistics("mean", grouped=False)
        assert summary.single
        assert summary.output_names == ["mean"]

    def test_normalize_statistics__defaults(self) -> None:
        """Checks that default global and grouped requests contain their expected statistics and display names."""

        # Normalize the default request for global output, grouped output, and a masked global calculation
        summary = _normalize_statistics(None, grouped=False)
        grouped = _normalize_statistics(None, grouped=True)
        masked = _normalize_statistics("all", grouped=False, masked=True)

        # Global defaults use display names, while grouped columns keep the names requested by stats()
        assert summary.names == [
            "Min",
            "Max",
            "Mean",
            "Standard deviation",
            "Valid count",
            "Total count",
            "Percentage valid points",
        ]
        assert grouped.output_names == [
            "count",
            "min",
            "max",
            "mean",
            "std",
            "totalcount",
            "percentagevalidpoints",
        ]
        assert masked.aliases[-4:] == [
            "validinliercount",
            "totalinliercount",
            "percentagevalidinlierpoints",
            "percentageinlierpoints",
        ]

    def test_reduce_values__global(self) -> None:
        """Checks basic global statistics for eager arrays with separate nodata values."""

        # Give two values different nodata locations on the same four positions
        first = np.array([1.0, np.nan, 3.0, 4.0])
        second = np.array([10.0, 20.0, np.nan, 40.0])
        statistics = _normalize_statistics(
            ["mean", "std", "sum", "validcount", "totalcount", "percentagevalidpoints"], grouped=False
        )

        # Reduce the complete arrays as one implicit group
        table, strategy = _reduce_values([first, second], statistics)

        # Check each value independently against NumPy and the known input size
        assert strategy == "dense"
        assert table.index.tolist() == [0]
        for index, values in enumerate([first, second]):
            finite = values[np.isfinite(values)]
            assert table.loc[0, (index, "count")] == finite.size
            assert table.loc[0, (index, "mean")] == pytest.approx(finite.mean())
            assert table.loc[0, (index, "std")] == pytest.approx(finite.std())
            assert table.loc[0, (index, "sum")] == pytest.approx(finite.sum())
            assert table.loc[0, (index, "validcount")] == finite.size
            assert table.loc[0, (index, "totalcount")] == values.size
            assert table.loc[0, (index, "percentagevalidpoints")] == 75

    def test_reduce_values__groups(self) -> None:
        """Checks grouped eager statistics from integer group IDs, including excluded and nodata locations."""

        # Use three groups, one excluded position (doesn't belong to any group), and a
        # different nodata location for first/second values
        group_ids = np.array([[0, 1, 0, 1], [2, 2, -1, 1]])
        first = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 7.0, 8.0, 10.0]])
        second = np.array([[10.0, np.nan, 30.0, 40.0], [50.0, 70.0, 80.0, 100.0]])
        statistics = _normalize_statistics(
            ["mean", "std", "sum", "min", "max", "rmse", "validcount", "totalcount", "percentagevalidpoints"]
        )

        # Calculate all groups directly from their integer IDs
        table, strategy = _reduce_values(
            [first, second], statistics, group_ids=group_ids, total_groups=4, strategy="auto"
        )

        # Compare every observed group and value with the corresponding NumPy calculation
        assert strategy == "dense"
        assert table.index.tolist() == [0, 1, 2]
        for index, values in enumerate([first, second]):
            for group_id in table.index:
                members = values[group_ids == group_id]
                finite = members[np.isfinite(members)]
                expected = [
                    finite.size,
                    finite.mean(),
                    finite.std(),
                    finite.sum(),
                    finite.min(),
                    finite.max(),
                    np.sqrt(np.mean(finite**2)),
                    finite.size,
                    members.size,
                    100 * finite.size / members.size,
                ]
                np.testing.assert_allclose(table.loc[group_id, index], expected)

    def test_reduce_values__complete_group_values(self) -> None:
        """Checks that medians, NMAD and custom functions use all values at once from each eager group."""

        # Change group IDs so each group is noncontiguous, then add NaNs to distinguish valid/total count
        values = np.arange(18, dtype=float)
        values[::5] = np.nan
        group_ids = np.arange(values.size) % 3
        statistics = _normalize_statistics(["median", "nmad", np.size])

        # Exact statistics use the complete values from each group
        table, strategy = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3, strategy="auto")

        # Check all results directly from the original groups
        assert strategy == "groupwise"
        for group_id in range(3):
            members = values[group_ids == group_id]
            median = np.nanmedian(members)
            expected = [
                np.isfinite(members).sum(),
                median,
                1.4826 * np.nanmedian(np.abs(members - median)),
                members.size,
            ]
            np.testing.assert_allclose(table.loc[group_id, 0], expected)

    @pytest.mark.parametrize("strategy", ["dense", "sparse", "groupwise"])
    @pytest.mark.parametrize("unsigned", [False, True])
    def test_reduce_values__large_integer_extrema(self, strategy: str, unsigned: bool) -> None:
        """Checks that eager grouped extrema remain exact beyond floating-point integer precision."""

        # Neighboring values above 2**53 would become indistinguishable in a floating-point intermediate array
        dtype = np.uint64 if unsigned else np.int64
        offset = 2**63 + 10 if unsigned else 2**60
        values = np.array([offset + step for step in [1, 3, 5, 7, 9, 11]], dtype=dtype)
        source = np.ma.array(values, mask=[False, True, False, False, True, True])
        group_ids = np.array([0, 0, 1, 1, 2, 2])
        names = ["min", "max", "median" if strategy == "groupwise" else "mean"]
        statistics = _normalize_statistics(names)

        # Reduce two populated groups and one group containing only nodata values
        table, _ = _reduce_values([source], statistics, group_ids=group_ids, total_groups=4, strategy=strategy)

        # Compare extrema as integers so floating-point rounding cannot hide an incorrect result
        assert int(table.loc[0, (0, "min")]) == int(values[0])
        assert int(table.loc[0, (0, "max")]) == int(values[0])
        assert int(table.loc[1, (0, "min")]) == int(values[2])
        assert int(table.loc[1, (0, "max")]) == int(values[3])
        assert pd.isna(table.loc[2, (0, "min")])
        assert pd.isna(table.loc[2, (0, "max")])
        assert np.array_equal(table[(0, "count")], [1, 2, 0])

    def test_reduce_global_values__output_forms(self) -> None:
        """
        Checks that global reduction restores scalar output for one scalar input (e.g. asking for just the "mean"
        returns a single value, not a dictionary)."""

        # Prepare the same unmasked arrays in the form returned by global selection
        first = np.array([1.0, 2.0, np.nan, 4.0])
        second = first + 10
        selected = {"first": (first, None), "second": (second, None)}

        # One statistic returns scalar values, while a list returns a dictionary for each selected value
        single = _reduce_global_values(
            {"first": selected["first"]}, _normalize_statistics("mean", grouped=False), strategy="auto", mp_config=None
        )
        multiple = _reduce_global_values(
            selected,
            _normalize_statistics(["mean", "validcount", "totalcount"], grouped=False),
            strategy="auto",
            mp_config=None,
        )

        # Check both established global output forms
        assert single == pytest.approx(7 / 3)
        assert multiple == {
            "first": {"mean": pytest.approx(7 / 3), "validcount": 3, "totalcount": 4},
            "second": {"mean": pytest.approx(37 / 3), "validcount": 3, "totalcount": 4},
        }


class TestReductionChunked:
    """
    Tests reductions split across Dask chunks or Multiproc tiles.

    Backend results are compared exactly with eager calculations, and Dask inputs have to remain lazy and file inputs
    remain unloaded.
    Additional tests cover notably the calculation strategies:
    - "auto" selects one of the strategies below from the requested statistics and number of groups.
    - "dense" includes every possible group in the summary from each chunk.
    - "sparse" includes only the groups found in each chunk.
    - "groupwise" gathers all values from each group before calculating its statistics (required for median/NMAD/etc).
    """

    @pytest.mark.parametrize("strategy", ["auto", "dense", "sparse", "groupwise"])
    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_reduce_values__backend_equality(self, strategy: str, backend: str) -> None:
        """Checks that every chunk strategy matches eager results for Dask and Multiproc inputs."""

        # Spread three groups and separate nodata patterns across a rectangular array
        shape = (9, 10)
        group_ids: Any = (np.arange(np.prod(shape)) % 3).reshape(shape)
        first = np.arange(np.prod(shape), dtype=float).reshape(shape)
        second = first * 2 + 10
        first.flat[::7] = np.nan
        second.flat[::11] = np.inf
        group_ids.flat[::13] = -1
        values: list[Any] = [first, second]
        statistics = _normalize_statistics(["mean", "std", "sum", "min", "max", "rmse", "totalcount"])
        expected, _ = _reduce_values(values, statistics, group_ids=group_ids, total_groups=4, strategy=strategy)

        # Store the same values in Dask chunks or ask Multiproc to read NumPy tiles
        config = None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            values = [da.from_array(first, chunks=(2, 4)), da.from_array(second, chunks=(3, 2))]
            group_ids = da.from_array(group_ids, chunks=(3, 3))
        else:
            config = MultiprocConfig(chunks=(2, 4))

        # Check the computed result and confirm that Dask inputs are still lazy collections
        result, resolved = _reduce_values(
            values,
            statistics,
            group_ids=group_ids,
            total_groups=4,
            strategy=strategy,
            mp_config=config,
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected)
        assert resolved == ("dense" if strategy == "auto" else strategy)
        if backend == "dask":
            assert all(isinstance(value, da.Array) for value in values)
            assert isinstance(group_ids, da.Array)

    def test_reduce_values__multiproc_reader_loading(self, tmp_path: Path) -> None:
        """Checks that Multiproc reduces raster blocks without loading the complete source."""

        # Write a raster file to disk with nodata values spread across several reduction tiles
        values = np.arange(35, dtype=float).reshape(5, 7)
        values[1, 2] = np.nan
        values[3, 5] = np.nan
        path = tmp_path / "reduction.tif"
        gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631, nodata=np.nan).to_file(path)
        source = gu.Raster(path)
        reader = _ValueReader(source, selector=1)
        group_ids = np.indices(values.shape).sum(axis=0) % 3
        statistics = _normalize_statistics(["mean", "std", "sum", "min", "max"])
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3)

        # Read only raster windows in worker processes and return a computed table
        from geoutils.multiproc.cluster import MpCluster

        with MpCluster({"nb_workers": 2}) as cluster:
            result, _ = _reduce_values(
                [reader],
                statistics,
                group_ids=group_ids,
                total_groups=3,
                mp_config=MultiprocConfig(chunks=(2, 3), cluster=cluster),
            )

        # Compare every result while the original Raster still has no loaded data
        pd.testing.assert_frame_equal(result, expected)
        assert not source.is_loaded

    @pytest.mark.parametrize("dense", [True, False])
    def test_reduce_block__dense_and_sparse(self, dense: bool) -> None:
        """Checks that block summaries merge into the same table as a complete eager reduction."""

        # Divide two values and three observed group IDs between two blocks
        group_ids = np.array([0, 2, -1, 0, 1, 2, 1, 2])
        first = np.array([1.0, 2.0, 30.0, np.nan, 5.0, 7.0, 9.0, 11.0])
        second = np.array([10.0, np.nan, 300.0, 40.0, 50.0, 70.0, 90.0, 110.0])
        statistics = _normalize_statistics(["mean", "std", "sum", "min", "max", "rmse", "validcount", "totalcount"])
        aliases = {alias for alias in statistics.aliases if alias is not None}

        # Summarize each block, merge their small arrays, and construct the final table
        summaries = [
            _reduce_block([first[block], second[block]], group_ids[block], 4, dense, aliases)
            for block in [slice(0, 4), slice(4, 8)]
        ]
        summary = _merge_blocks(summaries)
        result = _finalize_blocks(summary, statistics)
        expected, _ = _reduce_values([first, second], statistics, group_ids=group_ids, total_groups=4, strategy="dense")

        # Dense blocks reserve every group; sparse blocks store only the IDs present in each block
        expected_labels = np.arange(4) if dense else np.array([0, 2])
        assert np.array_equal(summaries[0][0], expected_labels)
        assert set(summaries[0][2]) == {"count", "mean", "m2", "sum", "min", "max", "sumofsquares"}
        pd.testing.assert_frame_equal(result, expected)

    def test_aggregate_collected__complete_groups(self) -> None:
        """Checks that values collected from separate blocks reconstruct complete groups in their original order."""

        # Split two interleaved groups and one nodata value across three blocks
        group_ids = np.array([0, 1, 0, 1, 0, 1, 0, 1, -1])
        values = np.array([9.0, 2.0, np.nan, 4.0, 5.0, 6.0, 3.0, 8.0, 100.0])
        statistics = _normalize_statistics(["median", "nmad", np.size])

        # Select both groups from each block, then calculate from their joined values
        blocks = [
            _collect_block([values[block]], group_ids[block], [0, 1])
            for block in [slice(0, 3), slice(3, 6), slice(6, 9)]
        ]
        result = _aggregate_collected(blocks, statistics)
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=2, strategy="groupwise")

        # The excluded final value is absent, while group order and nodata values are preserved
        assert np.array_equal(np.concatenate([block[0] for block in blocks]), group_ids[group_ids >= 0])
        pd.testing.assert_frame_equal(result, expected, check_exact=True)

    @pytest.mark.parametrize(
        "aliases,total_groups,expected,mergeable",
        [
            (["mean"], 4096, "dense", True),
            (["mean"], 4097, "sparse", True),
            (["median"], 2, "groupwise", False),
            ([None], 2, "groupwise", False),
        ],
    )
    def test_resolve_strategy__automatic(
        self, aliases: list[str | None], total_groups: int, expected: str, mergeable: bool
    ) -> None:
        """Checks that auto selects a strategy from the statistics and number of groups."""

        strategy, can_merge = _resolve_strategy(aliases, "auto", total_groups, chunked=True)
        assert strategy == expected
        assert can_merge is mergeable

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_reduce_values__groupwise_backends(self, backend: str) -> None:
        """Checks that chunked medians, NMAD and custom functions exactly match complete eager groups."""

        # Spread each group and its nodata values across several chunks
        values: Any = np.arange(35, dtype=float)
        values[::6] = np.nan
        group_ids: Any = np.arange(35) % 4
        statistics = _normalize_statistics(["median", "nmad", np.size])
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=4, strategy="groupwise")
        config = MultiprocConfig(chunks=6) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            values = da.from_array(values, chunks=6)
            group_ids = da.from_array(group_ids, chunks=5)

        # Gather complete groups through the selected backend
        result, strategy = _reduce_values(
            [values],
            statistics,
            group_ids=group_ids,
            total_groups=4,
            strategy="auto",
            mp_config=config,
        )

        # The result is computed, exactly equal, and leaves Dask inputs as lazy arrays
        assert strategy == "groupwise"
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        if backend == "dask":
            assert isinstance(values, da.Array)
            assert isinstance(group_ids, da.Array)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("masked", [False, True])
    def test_reduce_values__integer_squares(self, backend: str, masked: bool) -> None:
        """Checks that chunked integer squares exactly match the eager floating-point calculation."""

        # Squared elevations exceed int16, with one optional location excluded by a NumPy mask
        values = np.array([[2000, 3000], [5000, 4000]], dtype=np.int16)
        source: Any = np.ma.array(values, mask=[[False, True], [False, False]]) if masked else values
        statistics = _normalize_statistics(["sumofsquares", "rmse"], grouped=False)
        expected, _ = _reduce_values([source], statistics)
        config = MultiprocConfig(chunks=(1, 2)) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            source = da.from_array(source, chunks=(1, 2))

        # Reduce separate blocks without squaring in the original integer data type
        result, _ = _reduce_values([source], statistics, mp_config=config)
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        if backend == "dask":
            assert isinstance(source, da.Array)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("strategy", ["dense", "sparse", "groupwise"])
    @pytest.mark.parametrize("unsigned", [False, True])
    def test_reduce_values__large_integer_extrema(self, backend: str, strategy: str, unsigned: bool) -> None:
        """Checks that chunked extrema remain exact for signed and unsigned integers beyond float precision."""

        # Include two populated groups and one group containing only masked integer values
        dtype = np.uint64 if unsigned else np.int64
        offset = 2**63 + 10 if unsigned else 2**60
        values = np.array([offset + step for step in [1, 3, 5, 7, 9, 11]], dtype=dtype)
        source: Any = np.ma.array(values, mask=[False, True, False, False, True, True])
        group_ids: Any = np.array([0, 0, 1, 1, 2, 2])
        names = ["min", "max", "median" if strategy == "groupwise" else "mean"]
        statistics = _normalize_statistics(names)
        expected, _ = _reduce_values([source], statistics, group_ids=group_ids, total_groups=4, strategy=strategy)
        config = MultiprocConfig(chunks=2) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            source = da.from_array(source, chunks=2)
            group_ids = da.from_array(group_ids, chunks=3)

        # Compare the complete chunked table and exact extrema with the eager reducer
        result, _ = _reduce_values(
            [source],
            statistics,
            group_ids=group_ids,
            total_groups=4,
            strategy=strategy,
            mp_config=config,
        )
        pd.testing.assert_frame_equal(result, expected)
        assert int(result.loc[0, (0, "min")]) == int(values[0])
        assert int(result.loc[1, (0, "max")]) == int(values[3])
        assert pd.isna(result.loc[2, (0, "min")])
        if backend == "dask":
            assert isinstance(source, da.Array)
            assert isinstance(group_ids, da.Array)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("nodata", ["empty", "nan", "masked"])
    def test_reduce_values__empty_sums(self, backend: str, nodata: str) -> None:
        """Checks that chunked sums are undefined when empty, NaN or masked inputs have no valid values."""

        # Build the requested empty representation and its eager reference
        source: Any = np.empty(0) if nodata == "empty" else np.full(4, np.nan)
        if nodata == "masked":
            source = np.ma.array(np.arange(4), mask=True)
        statistics = _normalize_statistics(["sum", "sumofsquares", "validcount"], grouped=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Empty raster")
            expected, _ = _reduce_values([source], statistics)
        config = MultiprocConfig(chunks=2) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            source = da.from_array(source, chunks=2)

        # Keep undefined sums and a zero valid count after reducing separate chunks
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Empty raster")
            result, _ = _reduce_values([source], statistics, mp_config=config)
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        assert np.isnan(result.loc[0, (0, "sum")])
        assert result.loc[0, (0, "validcount")] == 0
        if backend == "dask":
            assert isinstance(source, da.Array)

    @pytest.mark.parametrize("strategy", ["dense", "sparse"])
    def test_reduce_values__variance_with_large_offset(self, strategy: str) -> None:
        """Checks that merging Dask chunks preserves small variation around a large base value."""

        # Values near 1e8 expose unstable variance formulas that subtract two large squared totals
        da = pytest.importorskip("dask.array")
        values = 1e8 + np.random.default_rng(42).normal(scale=0.1, size=3000)
        group_ids = np.arange(values.size) % 3
        statistics = _normalize_statistics("std")
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3, strategy=strategy)

        # Compare chunked pairwise variance with the eager reducer and NumPy
        result, _ = _reduce_values(
            [da.from_array(values, chunks=101)],
            statistics,
            group_ids=group_ids,
            total_groups=3,
            strategy=strategy,
        )
        pd.testing.assert_frame_equal(result, expected)
        direct = [np.std(values[group_ids == group_id]) for group_id in range(3)]
        np.testing.assert_allclose(result[(0, "std")], direct)

    def test_reduce_values__groupwise_block_reads(self) -> None:
        """Checks that groupwise Dask reduction reads only blocks containing a requested group."""

        # Record value blocks read after group locations are known and reject the excluded final block
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        reads = []

        def read_values(block: int) -> NDArrayNum:
            """Record each value block read and reject the block outside every group."""

            assert block < 2
            reads.append(block)
            return np.arange(4, dtype=float) + 4 * block

        blocks = [da.from_delayed(dask.delayed(read_values)(block), shape=(4,), dtype=float) for block in range(3)]
        values = da.concatenate(blocks)
        group_ids = np.array([0, 1, 0, 1, 2, 2, 2, 2, -1, -1, -1, -1])
        statistics = _normalize_statistics("median")

        # Calculate medians from only the blocks containing the three groups
        with dask.config.set(scheduler="synchronous"):
            result, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3, strategy="groupwise")

        # Every needed block is read once and the excluded block is never loaded
        assert sorted(reads) == [0, 1]
        np.testing.assert_allclose(result[(0, "median")], [1.0, 2.0, 5.5])

    def test_statistics_dask__matches_shared_reducer(self) -> None:
        """Checks that the optional native Dask reductions are lazy and match the shared GeoUtils reducer."""

        # Include nodata values across uneven chunks and request every numerical estimator
        import_optional("dask")
        import dask
        import dask.array as da

        values = np.arange(35, dtype=float).reshape(5, 7)
        values[1, 2] = np.nan
        values[3, 5] = np.nan
        source = da.from_array(values, chunks=(2, 3))
        aliases = {
            "mean",
            "median",
            "min",
            "max",
            "sum",
            "sumofsquares",
            "90thpercentile",
            "iqr",
            "le90",
            "nmad",
            "rmse",
            "std",
        }
        statistics = _normalize_statistics(sorted(aliases), grouped=False)
        expected, _ = _reduce_values([values], statistics)

        # Build both Dask calculations before computing their small results
        native, native_count = _statistics_dask(source, aliases)
        assert all(isinstance(value, da.Array) for value in native.values())
        assert isinstance(native_count, da.Array)
        shared, _ = _reduce_values([source], statistics)
        native_values, count = dask.compute(native, native_count)

        # Match both implementations with the eager reducer while the input remains a Dask array
        pd.testing.assert_frame_equal(shared, expected)
        assert count == expected.loc[0, (0, "count")]
        for name, value in native_values.items():
            assert value == pytest.approx(expected.loc[0, (0, name)])
        assert isinstance(source, da.Array)

    @pytest.mark.parametrize("strategy", ["dense", "sparse", "groupwise"])
    def test_reduce_values__multiproc_workers(self, strategy: str) -> None:
        """Checks that real Multiproc workers return the same grouped table as the eager reducer."""

        # Split a rectangular array so every group crosses several worker tiles
        from geoutils.multiproc.cluster import MpCluster

        shape = (31, 47)
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        group_ids = np.indices(shape).sum(axis=0) % 3
        statistics = _normalize_statistics("mean")
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3, strategy=strategy)

        # Calculate the same groups with two worker processes
        with MpCluster({"nb_workers": 2}) as cluster:
            result, _ = _reduce_values(
                [values],
                statistics,
                group_ids=group_ids,
                total_groups=3,
                strategy=strategy,
                mp_config=MultiprocConfig(chunks=(7, 11), cluster=cluster),
            )

        # Check the complete computed table
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "shape,chunks,layout",
        [
            ((131, 197), (37, 61), "interleaved"),
            ((257, 509), (128, 193), "local"),
            ((131, 197), (2048, 2048), "interleaved"),
        ],
    )
    @pytest.mark.parametrize("strategy", ["dense", "sparse"])
    def test_reduce_values__chunk_layouts(
        self, shape: tuple[int, int], chunks: tuple[int, int], layout: str, strategy: str
    ) -> None:
        """Checks mergeable statistics across local, interleaved, uneven and oversized Dask chunks."""

        # Create exact quarter-step values with separate nodata patterns
        da = pytest.importorskip("dask.array")
        rows, columns = np.indices(shape)
        positions = rows * shape[1] + columns
        first = 20 + (positions % 97) * 0.25
        second = -2 * first + positions % 3
        first[positions % 17 == 0] = np.nan
        second[positions % 29 == 0] = np.inf

        # Place groups in separate areas or spread them across the array, then exclude selected locations
        group_ids = positions % 8
        if layout == "local":
            group_ids = (rows * 2 // shape[0]) * 4 + columns * 4 // shape[1]
        keep = (rows + columns) % 19 != 0
        keep[: shape[0] // 4, : shape[1] // 4] = False
        group_ids[~keep | (positions % 31 == 0)] = -1
        statistics = _normalize_statistics(["mean", "std", "sum", "min", "max", "rmse", "totalcount"])
        expected, _ = _reduce_values(
            [first, second], statistics, group_ids=group_ids, total_groups=9, strategy=strategy
        )

        # Give values and group IDs different chunk layouts
        values = [da.from_array(first, chunks=chunks), da.from_array(second, chunks=chunks[::-1])]
        lazy_ids = da.from_array(group_ids, chunks=(chunks[0] + 3, chunks[1] + 5))
        result, _ = _reduce_values(values, statistics, group_ids=lazy_ids, total_groups=9, strategy=strategy)

        # Compare every computed column and confirm that all inputs remain Dask arrays
        pd.testing.assert_frame_equal(result, expected)
        assert all(isinstance(value, da.Array) for value in values)
        assert isinstance(lazy_ids, da.Array)

    def test_reduce_values__sparse_group_ids(self) -> None:
        """Checks that automatic sparse reduction keeps widely separated group IDs distinct."""

        # Use three group IDs from a space just above the automatic dense threshold
        da = pytest.importorskip("dask.array")
        total_groups = 4097
        labels = np.array([0, total_groups // 2, total_groups - 1])
        values = np.arange(30, dtype=float)
        group_ids = labels[np.arange(values.size) % 3]
        statistics = _normalize_statistics(["mean", "std"])

        # Let auto select sparse summaries for the Dask chunks
        result, strategy = _reduce_values(
            [da.from_array(values, chunks=7)],
            statistics,
            group_ids=da.from_array(group_ids, chunks=5),
            total_groups=total_groups,
            strategy="auto",
        )

        # Check the selected strategy, exact group IDs, and values from each complete group
        assert strategy == "sparse"
        assert result.index.tolist() == labels.tolist()
        for label in labels:
            members = values[group_ids == label]
            np.testing.assert_allclose(result.loc[label], [members.size, members.mean(), members.std()])


class TestReductionErrors:
    """Test module for validation errors and warnings raised by statistic reduction."""

    def test_normalize_statistics__error_names(self) -> None:
        """Checks that invalid statistic names are rejected."""

        # Two partial functions will have the same output name even though their percentile arguments differ
        percentiles = [partial(np.nanpercentile, q=25), partial(np.nanpercentile, q=75)]
        with pytest.raises(ValueError, match="unique"):
            _normalize_statistics(percentiles)

        # Grouped output reserves count for a callable and rejects names without a known reducer
        def count(values: Any) -> int:
            return len(values)

        # This needs to fails because "count" is a reserved name
        with pytest.raises(ValueError, match="reserved"):
            _normalize_statistics([count])
        # Wrong input name
        with pytest.raises(ValueError, match="Unknown statistic names"):
            _normalize_statistics(["made_up"])
        # The "all" statistics input can not be combined with others
        with pytest.raises(ValueError, match="cannot be combined"):
            _normalize_statistics(["all", "mean"])

        # Every request must be either a recognized name or a callable function
        with pytest.raises(TypeError, match="names or callable"):
            _normalize_statistics(cast(Any, [object()]))

    def test_resolve_strategy__error_invalid(self) -> None:
        """Checks that unknown strategies and incomplete-group calculations are rejected for chunked inputs."""

        # Reject an unknown option independently of the requested statistics
        with pytest.raises(ValueError, match="must be 'auto', 'dense', 'sparse' or 'groupwise'"):
            _resolve_strategy(["mean"], "topk", 2, chunked=True)

        # Median and custom functions need complete groups rather than dense or sparse summaries
        with pytest.raises(ValueError, match="require ``strategy``='groupwise'"):
            _resolve_strategy(["median"], "dense", 2, chunked=True)
        with pytest.raises(ValueError, match="require ``strategy``='groupwise'"):
            _resolve_strategy([None], "sparse", 2, chunked=True)

    @pytest.mark.parametrize("nodata", ["empty", "nan", "masked"])
    def test_reduce_values__empty_sums(self, nodata: str) -> None:
        """Checks that eager sums warn and stay undefined when no valid values remain."""

        # Test an empty array, explicit NaNs, and values excluded by a NumPy masked array
        source: Any = np.empty(0) if nodata == "empty" else np.full(4, np.nan)
        if nodata == "masked":
            source = np.ma.array(np.arange(4), mask=True)
        statistics = _normalize_statistics(["sum", "sumofsquares", "validcount"], grouped=False)

        # A valid count of zero must be distinguishable from a real sum of zeros
        with pytest.warns(UserWarning, match="Empty raster"):
            table, _ = _reduce_values([source], statistics)
        assert np.isnan(table.loc[0, (0, "sum")])
        assert np.isnan(table.loc[0, (0, "sumofsquares")])
        assert table.loc[0, (0, "validcount")] == 0
