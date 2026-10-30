"""Tests shared global and grouped statistic reductions."""

from __future__ import annotations

import pickle
import warnings
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from affine import Affine

import geoutils as gu
from geoutils._dispatch import is_dask_array
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.multiproc.readers import _ValueReader
from geoutils.stats.reduction import (
    _aggregate_collected,
    _collect_block,
    _digest_quantile,
    _finalize_blocks,
    _flox_groupby_reduce,
    _groupby_reduce_flox,
    _groupby_reduce_geoutils,
    _merge_blocks,
    _normalize_statistics,
    _reduce_block,
    _reduce_global_values,
    _reduce_values,
    _resolve_reduction_method,
    _Statistics,
    _tdigest_aggregation,
)


class TestReduction:
    """
    Tests the reduction (i.e., applying statistical reducer function like mean() or std()) for in-memory arrays.

    The Dask/Multiproc chunked execution are tested further below in TestReductionChunked, and errors/warnings in
    TestReductionErrors.

    Flox/GeoUtils options, and Crick t-digest support are tested in all of those.

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
        """Checks basic global statistics for in-memory arrays with separate nodata values."""

        # Give two values different nodata locations on the same four positions
        first = np.array([1.0, np.nan, 3.0, 4.0])
        second = np.array([10.0, 20.0, np.nan, 40.0])
        statistics = _normalize_statistics(
            ["mean", "std", "sum", "validcount", "totalcount", "percentagevalidpoints"], grouped=False
        )

        # Reduce the complete arrays as one implicit group
        table, method = _reduce_values([first, second], statistics)

        # Check each value independently against NumPy and the known input size
        assert method == "map-reduce"
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
        """Checks grouped in-memory statistics from integer group IDs, including excluded and nodata locations."""

        # Use three groups, one excluded position (doesn't belong to any group), and a
        # different nodata location for first/second values
        group_ids = np.array([[0, 1, 0, 1], [2, 2, -1, 1]])
        first = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 7.0, 8.0, 10.0]])
        second = np.array([[10.0, np.nan, 30.0, 40.0], [50.0, 70.0, 80.0, 100.0]])
        statistics = _normalize_statistics(
            ["mean", "std", "sum", "min", "max", "rmse", "validcount", "totalcount", "percentagevalidpoints"]
        )

        # Calculate all groups directly from their integer IDs
        table, method = _reduce_values([first, second], statistics, group_ids=group_ids, total_groups=4, method="auto")

        # Compare every observed group and value with the corresponding NumPy calculation
        assert method == "map-reduce"
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

    @pytest.mark.parametrize("backend,engine", [("geoutils", None), ("flox", None), ("flox", "numba")])
    def test_groupby_reduce__per_value_reductions(self, backend: str, engine: str | None) -> None:
        """Checks that each value array can request different reductions."""

        if engine == "numba":
            pytest.importorskip("numba")

        if backend == "flox":
            pytest.importorskip("flox")

        # Create two arrays with excluded values and an empty group
        distances = np.array([1.0, 2.0, 3.0, 4.0, 99.0])
        differences = np.array([2.0, 4.0, 6.0, 8.0, 99.0])
        groups = np.array([0, 1, 0, 1, -1])
        reductions = [{"sum"}, {"sumofsquares", "min", "max", "std"}]

        # Request different reductions for each array
        reduce = _groupby_reduce_geoutils if backend == "geoutils" else _groupby_reduce_flox
        labels, size, state = reduce([distances, differences], groups, 3, reductions, engine=engine)

        # Check results and count dtype
        np.testing.assert_array_equal(labels, [0, 1, 2])
        np.testing.assert_array_equal(size, [2, 2, 0])
        np.testing.assert_array_equal(state["count"], [[2, 2, 0], [2, 2, 0]])
        assert state["count"].dtype == np.int64
        np.testing.assert_array_equal(state["sum"][0], [4, 6, np.nan])
        np.testing.assert_array_equal(state["sumofsquares"][1], [40, 80, np.nan])
        np.testing.assert_array_equal(state["min"][1], [2, 4, np.nan])
        np.testing.assert_array_equal(state["max"][1], [6, 8, np.nan])
        np.testing.assert_array_equal(state["std"][1], [2, 2, np.nan])
        assert np.isnan(state["sum"][1]).all()
        assert "mean" not in state and "m2" not in state

    def test_merge_blocks__integer_counts(self) -> None:
        """Checks that merging large integer counts preserves precision."""

        # Create summary counts above the exact float64 integer range
        counts = np.array([[2**53, 0], [1, 3], [2, 4]], dtype=np.int64)
        distance_sums = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        estimator_sums = np.array([[2, 4], [6, 8], [10, 12]], dtype=np.float64)
        summaries = []
        for count, distance, estimator in zip(counts, distance_sums, estimator_sums):
            state = {"count": count[None, :], "distance": distance[None, :], "estimator": estimator[None, :]}
            summaries.append((np.arange(2), count, state))

        # Merge summaries and check precision
        _, _, result = _merge_blocks(summaries)
        np.testing.assert_array_equal(result["count"][0], [2**53 + 3, 7])
        np.testing.assert_array_equal(result["distance"][0], [9, 12])
        np.testing.assert_array_equal(result["estimator"][0], [18, 24])
        assert result["count"].dtype == np.int64
        assert result["distance"].dtype == np.float64
        assert result["estimator"].dtype == np.float64

    def test_reduce_values__complete_group_values(self) -> None:
        """Checks that medians, NMAD and custom functions use all values at once from each in-memory group."""

        # Change group IDs so each group is noncontiguous, then add NaNs to distinguish valid/total count
        values = np.arange(18, dtype=float)
        values[::5] = np.nan
        group_ids = np.arange(values.size) % 3
        statistics = _normalize_statistics(["median", "nmad", np.size])

        # Exact statistics use the complete values from each group
        table, method = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3, method="auto")

        # Check all results directly from the original groups
        assert method == "groupwise"
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

    @pytest.mark.parametrize("method,reindex", [("map-reduce", True), ("map-reduce", False), ("groupwise", None)])
    @pytest.mark.parametrize("unsigned", [False, True])
    def test_reduce_values__large_integer_extrema(self, method: str, reindex: bool | None, unsigned: bool) -> None:
        """Checks that in-memory grouped extrema remain exact beyond floating-point integer precision."""

        # Neighboring values above 2**53 would become indistinguishable in a floating-point intermediate array
        dtype = np.uint64 if unsigned else np.int64
        offset = 2**63 + 10 if unsigned else 2**60
        values = np.array([offset + step for step in [1, 3, 5, 7, 9, 11]], dtype=dtype)
        source = np.ma.array(values, mask=[False, True, False, False, True, True])
        group_ids = np.array([0, 0, 1, 1, 2, 2])
        names = ["min", "max", "median" if method == "groupwise" else "mean"]
        statistics = _normalize_statistics(names)

        # Reduce two populated groups and one group containing only nodata values
        table, _ = _reduce_values(
            [source], statistics, group_ids=group_ids, total_groups=4, method=method, reindex=reindex
        )

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
            {"first": selected["first"]}, _normalize_statistics("mean", grouped=False), method="auto", mp_config=None
        )
        multiple = _reduce_global_values(
            selected,
            _normalize_statistics(["mean", "validcount", "totalcount"], grouped=False),
            method="auto",
            mp_config=None,
        )

        # Check both established global output forms
        assert single == pytest.approx(7 / 3)
        assert multiple == {
            "first": {"mean": pytest.approx(7 / 3), "validcount": 3, "totalcount": 4},
            "second": {"mean": pytest.approx(37 / 3), "validcount": 3, "totalcount": 4},
        }

    @pytest.mark.parametrize(
        "aliases,total_groups,expected_method,expected_reindex,mergeable",
        [
            (["mean"], 4096, "map-reduce", True, True),
            (["mean"], 4097, "map-reduce", False, True),
            (["median"], 2, "map-reduce", True, True),
            ([None], 2, "groupwise", None, False),
        ],
    )
    def test_resolve_reduction_method__automatic(
        self,
        aliases: list[str | None],
        total_groups: int,
        expected_method: str,
        expected_reindex: bool | None,
        mergeable: bool,
    ) -> None:
        """Checks that automatic method selection matches the requested statistics."""

        method, reindex, can_merge = _resolve_reduction_method(aliases, total_groups)
        assert method == expected_method
        assert reindex is expected_reindex
        assert can_merge is mergeable

    @pytest.mark.parametrize("engine", [None, "numpy", "numba"])
    def test_flox_groupby_reduce__multiple_groupings(self, engine: str | None) -> None:
        """Checks that Flox handles multiple grouping arrays."""

        if engine == "numba":
            pytest.importorskip("numba")

        # Create two value arrays and two grouping arrays
        pytest.importorskip("flox")
        values = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
        rows = np.broadcast_to(np.array([0, 2, 0, 2])[:, None], (4, 5))
        columns = np.broadcast_to(np.array([1, 3, -1, 1, 3]), (4, 5))
        expected_groups = [pd.Index([2, 0, 8]), pd.Index([3, 1, 9])]
        expected = np.zeros((2, 3, 3), dtype=np.float64)
        for row, row_label in enumerate(expected_groups[0]):
            for column, column_label in enumerate(expected_groups[1]):
                selected = (rows == row_label) & (columns == column_label)
                expected[:, row, column] = values[:, selected].sum(axis=1, dtype=np.float64)

        # Reduce both grouping arrays together
        result = _flox_groupby_reduce(
            values,
            [rows, columns],
            expected_groups,
            function="sum",
            use_dask=False,
            fill_value=0,
            dtype=np.float64,
            engine=engine,
        )
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float64


class TestReductionChunked:
    """
    Tests reductions split across Dask chunks or Multiproc tiles.

    Backend results are compared exactly with in-memory calculations, and Dask inputs have to remain lazy and file
    inputs remain unloaded.
    Additional tests cover notably the calculation strategies:
    - "auto" selects one of the strategies below from the requested statistics and number of groups.
    - "map-reduce" with reindex=True includes every possible group in the summary from each chunk.
    - "map-reduce" with reindex=False includes only the groups found in each chunk.
    - "groupwise" gathers all values from each group before calculating its statistics (required for median/NMAD/etc).
    """

    @pytest.mark.parametrize(
        "method,reindex", [("auto", None), ("map-reduce", True), ("map-reduce", False), ("groupwise", None)]
    )
    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_reduce_values__backend_equality(self, method: str, reindex: bool | None, backend: str) -> None:
        """Checks that every chunk strategy matches in-memory results for Dask and Multiproc inputs."""

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
        expected, _ = _reduce_values(
            values, statistics, group_ids=group_ids, total_groups=4, method=method, reindex=reindex
        )

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
            method=method,
            reindex=reindex,
            mp_config=config,
        )
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, expected)
        assert resolved == ("map-reduce" if method == "auto" else method)
        if backend == "dask":
            assert all(isinstance(value, da.Array) for value in values)
            assert isinstance(group_ids, da.Array)

    @pytest.mark.parametrize("split_every", [2, 3, 8])
    @pytest.mark.parametrize("with_quantiles", [False, True])
    def test_reduce_values__bounded_dask_batches(self, split_every: int, with_quantiles: bool) -> None:
        """Checks that Dask reads at most split_every chunks per batch and matches in-memory reductions."""

        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        if with_quantiles:
            pytest.importorskip("crick")
        from dask.callbacks import Callback

        # Use enough chunks for several batches and a shorter final chunk
        values = np.arange(35, dtype=float)
        statistics = _normalize_statistics(["sum", "mean", "median", "nmad"] if with_quantiles else ["sum", "mean"])
        expected, _ = _reduce_values([values], statistics, method="map-reduce")
        reads: list[Any] = []
        batch_starts: list[int] = []

        def read_chunk(block: Any, block_id: Any = None) -> Any:
            """Record each source chunk read without changing its values."""

            if block.size:
                reads.append(block_id)
            return block

        # Record source reads within each scheduled batch
        source = da.from_array(values, chunks=2).map_blocks(read_chunk, dtype=float)
        assert isinstance(source, da.Array) and not reads
        with dask.config.set(scheduler="synchronous"), Callback(start=lambda graph: batch_starts.append(len(reads))):
            result, _ = _reduce_values([source], statistics, method="map-reduce", split_every=split_every)

        # Both NMAD passes must respect the batch size
        batch_reads = np.diff([*batch_starts, len(reads)])
        assert len(batch_reads) > 1 and np.all(batch_reads <= split_every)
        assert len(reads) == source.npartitions * (2 if with_quantiles else 1)
        assert isinstance(source, da.Array)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("backend,execution", [("geoutils", "dask"), ("geoutils", "lazy_mp"), ("flox", "dask")])
    def test_groupby_reduce__shared_source_reads(self, backend: str, execution: str) -> None:
        """Checks that batched reductions share source reads."""

        if backend == "flox":
            pytest.importorskip("flox")
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        # Create constant groups with uneven chunks
        groups = np.arange(47) % 3
        values = np.stack((groups + 1.0, groups * 2.0 + 3))
        reductions = [{"sum"}, {"mean", "std", "min", "max"}]
        reduce = _groupby_reduce_geoutils if backend == "geoutils" else _groupby_reduce_flox
        expected = reduce(list(values), groups, 4, reductions)
        reads = []

        def read_chunk(block: Any, block_id: Any = None) -> Any:
            """Record one source chunk read without modifying its values."""

            if block.size:
                reads.append(block_id)
            return block

        def check_task(key: Any, result: Any, *args: Any) -> None:
            """Check that no reduction task assembles the complete source array."""

            if isinstance(result, np.ndarray):
                assert result.shape != values.shape

        # Reduce two slices of the same lazy array
        source = da.from_array(values, chunks=(2, 7)).map_blocks(read_chunk, dtype=float)
        config = MultiprocConfig(chunks=7) if execution == "lazy_mp" else None
        assert not reads
        with dask.config.set(scheduler="synchronous"), Callback(posttask=check_task):
            result = reduce([source[0], source[1]], groups, 4, reductions, mp_config=config)

        # Check results, loading and source reads
        assert isinstance(source, da.Array)
        assert len(reads) == 7
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[1], expected[1])
        assert result[2].keys() == expected[2].keys()
        for name, array in result[2].items():
            assert isinstance(array, np.ndarray)
            np.testing.assert_array_equal(array, expected[2][name])

    @pytest.mark.parametrize("with_quantiles", [False, True])
    def test_reduce_values__multiproc_reader_loading(
        self, tmp_path: Path, with_quantiles: bool, mp_cluster: MpCluster
    ) -> None:
        """Checks that Multiproc reduces raster blocks without loading the complete source."""

        names = ["mean", "std", "sum", "min", "max"]
        if with_quantiles:
            pytest.importorskip("crick")
            names += ["median", "nmad"]

        # Write a raster file to disk with NaNs in different blocks
        values = np.arange(35, dtype=float).reshape(5, 7)
        values[1, 2] = np.nan
        values[3, 5] = np.nan
        path = tmp_path / "reduction.tif"
        gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 5), 32631, nodata=np.nan).to_file(path)
        source = gu.Raster(path)
        reader = _ValueReader(source, selector=1)
        group_ids = np.indices(values.shape).sum(axis=0) % 3
        statistics = _normalize_statistics(names)
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3)

        # Read only raster windows in worker processes and return a computed table

        result, _ = _reduce_values(
            [reader],
            statistics,
            group_ids=group_ids,
            total_groups=3,
            mp_config=MultiprocConfig(chunks=(2, 3), cluster=mp_cluster),
        )

        # Compare every result while the original Raster still has no loaded data
        pd.testing.assert_frame_equal(result, expected)
        assert not source.is_loaded

    @pytest.mark.parametrize("reindex", [True, False])
    def test_reduce_block__reindex(self, reindex: bool) -> None:
        """Checks that dense and sparse summaries give the same result."""

        # Create two value arrays split across two blocks
        group_ids = np.array([0, 2, -1, 0, 1, 2, 1, 2])
        first = np.array([1.0, 2.0, 30.0, np.nan, 5.0, 7.0, 9.0, 11.0])
        second = np.array([10.0, np.nan, 300.0, 40.0, 50.0, 70.0, 90.0, 110.0])
        statistics = _normalize_statistics(["mean", "std", "sum", "min", "max", "rmse", "validcount", "totalcount"])
        aliases = {alias for alias in statistics.aliases if alias is not None}

        # Calculate and merge block summaries
        summaries = [
            _reduce_block(
                [first[block], second[block]], group_ids[block], total_groups=4, reindex=reindex, statistics=aliases
            )
            for block in [slice(0, 4), slice(4, 8)]
        ]
        summary = _merge_blocks(summaries)
        result = _finalize_blocks(summary, statistics)
        expected, _ = _reduce_values(
            [first, second], statistics, group_ids=group_ids, total_groups=4, method="map-reduce", reindex=True
        )

        # Check dense and sparse group indexes
        expected_labels = np.arange(4) if reindex else np.array([0, 2])
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
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=2, method="groupwise")

        # The excluded final value is absent, while group order and nodata values are preserved
        assert np.array_equal(np.concatenate([block[0] for block in blocks]), group_ids[group_ids >= 0])
        pd.testing.assert_frame_equal(result, expected, check_exact=True)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    def test_reduce_values__groupwise_backends(self, backend: str) -> None:
        """Checks that chunked medians, NMAD and custom functions exactly match complete groups."""

        # Spread each group and its nodata values across several chunks
        values: Any = np.arange(35, dtype=float)
        values[::6] = np.nan
        group_ids: Any = np.arange(35) % 4
        statistics = _normalize_statistics(["median", "nmad", np.size])
        expected, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=4, method="groupwise")
        config = MultiprocConfig(chunks=6) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")

            values = da.from_array(values, chunks=6)
            group_ids = da.from_array(group_ids, chunks=5)

        # Gather complete groups through the selected backend
        result, method = _reduce_values(
            [values],
            statistics,
            group_ids=group_ids,
            total_groups=4,
            method="auto",
            mp_config=config,
        )

        # The result is computed, exactly equal, and leaves Dask inputs as lazy arrays
        assert method == "groupwise"
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        if backend == "dask":
            assert isinstance(values, da.Array)
            assert isinstance(group_ids, da.Array)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("dtype", [np.int16, np.int64])
    def test_reduce_values__integer_squares(self, backend: str, masked: bool, dtype: Any) -> None:
        """Checks that chunked integer squares exactly match the in-memory floating-point calculation."""

        # Squared elevations exceed the input dtype, with one optional location excluded by a NumPy mask
        values = np.array([[2000, 3000], [5000, 4000]], dtype=dtype)
        if dtype == np.int64:
            values += 2**40
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
        selected = np.ma.compressed(source.compute() if backend == "dask" else source)
        expected_sum = float(sum(int(value) ** 2 for value in selected))
        np.testing.assert_allclose(result.loc[0, (0, "sumofsquares")], expected_sum)
        np.testing.assert_allclose(result.loc[0, (0, "rmse")], np.sqrt(expected_sum / len(selected)))
        if backend == "dask":
            assert isinstance(source, da.Array)

    @pytest.mark.parametrize("backend", ["dask", "multiproc"])
    @pytest.mark.parametrize("method,reindex", [("map-reduce", True), ("map-reduce", False), ("groupwise", None)])
    @pytest.mark.parametrize("unsigned", [False, True])
    def test_reduce_values__large_integer_extrema(
        self, backend: str, method: str, reindex: bool | None, unsigned: bool
    ) -> None:
        """Checks that chunked extrema remain exact for signed and unsigned integers beyond float precision."""

        # Include two populated groups and one group containing only masked integer values
        dtype = np.uint64 if unsigned else np.int64
        offset = 2**63 + 10 if unsigned else 2**60
        values = np.array([offset + step for step in [1, 3, 5, 7, 9, 11]], dtype=dtype)
        source: Any = np.ma.array(values, mask=[False, True, False, False, True, True])
        group_ids: Any = np.array([0, 0, 1, 1, 2, 2])
        names = ["min", "max", "median" if method == "groupwise" else "mean"]
        statistics = _normalize_statistics(names)
        expected, _ = _reduce_values(
            [source], statistics, group_ids=group_ids, total_groups=4, method=method, reindex=reindex
        )
        config = MultiprocConfig(chunks=2) if backend == "multiproc" else None
        if backend == "dask":
            da = pytest.importorskip("dask.array")

            source = da.from_array(source, chunks=2)
            group_ids = da.from_array(group_ids, chunks=3)

        # Compare the complete chunked table and exact extrema with the in-memory reducer
        result, _ = _reduce_values(
            [source],
            statistics,
            group_ids=group_ids,
            total_groups=4,
            method=method,
            reindex=reindex,
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

        # Build the requested empty representation and its in-memory reference
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

    @pytest.mark.parametrize("reindex", [True, False])
    def test_reduce_values__variance_with_large_offset(self, reindex: bool) -> None:
        """Checks that merging Dask chunks preserves small variation around a large base value."""

        # Values near 1e8 expose unstable variance formulas that subtract two large squared totals
        da = pytest.importorskip("dask.array")

        values = 1e8 + np.random.default_rng(42).normal(scale=0.1, size=3000)
        group_ids = np.arange(values.size) % 3
        statistics = _normalize_statistics("std")
        expected, _ = _reduce_values(
            [values], statistics, group_ids=group_ids, total_groups=3, method="map-reduce", reindex=reindex
        )

        # Compare chunked pairwise variance with the in-memory reducer and NumPy
        result, _ = _reduce_values(
            [da.from_array(values, chunks=101)],
            statistics,
            group_ids=group_ids,
            total_groups=3,
            method="map-reduce",
            reindex=reindex,
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
            result, _ = _reduce_values([values], statistics, group_ids=group_ids, total_groups=3, method="groupwise")

        # Every needed block is read once and the excluded block is never loaded
        assert sorted(reads) == [0, 1]
        np.testing.assert_allclose(result[(0, "median")], [1.0, 2.0, 5.5])

    @pytest.mark.parametrize("method,reindex", [("map-reduce", True), ("map-reduce", False), ("groupwise", None)])
    def test_reduce_values__multiproc_workers(self, method: str, reindex: bool | None, mp_cluster: MpCluster) -> None:
        """Checks that real Multiproc workers return the same grouped table as the in-memory reducer."""

        # Split a rectangular array so every group crosses several worker tiles

        shape = (31, 47)
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        group_ids = np.indices(shape).sum(axis=0) % 3
        statistics = _normalize_statistics("mean")
        expected, _ = _reduce_values(
            [values], statistics, group_ids=group_ids, total_groups=3, method=method, reindex=reindex
        )

        # Calculate the same groups with two worker processes
        result, _ = _reduce_values(
            [values],
            statistics,
            group_ids=group_ids,
            total_groups=3,
            method=method,
            reindex=reindex,
            mp_config=MultiprocConfig(chunks=(13, 19), cluster=mp_cluster),
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
    @pytest.mark.parametrize("reindex", [True, False])
    def test_reduce_values__chunk_layouts(
        self, shape: tuple[int, int], chunks: tuple[int, int], layout: str, reindex: bool
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
            [first, second], statistics, group_ids=group_ids, total_groups=9, method="map-reduce", reindex=reindex
        )

        # Give values and group IDs different chunk layouts
        values = [da.from_array(first, chunks=chunks), da.from_array(second, chunks=chunks[::-1])]
        lazy_ids = da.from_array(group_ids, chunks=(chunks[0] + 3, chunks[1] + 5))
        result, _ = _reduce_values(
            values, statistics, group_ids=lazy_ids, total_groups=9, method="map-reduce", reindex=reindex
        )

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
        result, method = _reduce_values(
            [da.from_array(values, chunks=7)],
            statistics,
            group_ids=da.from_array(group_ids, chunks=5),
            total_groups=total_groups,
            method="auto",
        )

        # Check the selected method, exact group IDs, and values from each complete group
        assert method == "map-reduce"
        assert result.index.tolist() == labels.tolist()
        for label in labels:
            members = values[group_ids == label]
            np.testing.assert_allclose(result.loc[label], [members.size, members.mean(), members.std()])

    @pytest.mark.parametrize("execution", ["dask", "mp"])
    def test_stats__sparse_nmad(self, execution: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that sparse NMAD allocates only for observed groups."""

        # Create two groups with widely separated IDs
        pytest.importorskip("crick")
        if execution == "dask":
            da = pytest.importorskip("dask.array")

        from geoutils.stats import reduction

        values = np.arange(41, dtype=float)
        groups = np.where(np.arange(41) % 2, 900_000, 0)
        groups[:7] = -1
        original_full = reduction.np.full

        # Fail before a full group median allocation
        def bounded_full(shape: Any, *args: Any, **kwargs: Any) -> Any:
            assert np.prod(shape) < 1000
            return original_full(shape, *args, **kwargs)

        # Run sparse NMAD with chunks
        source = da.from_array(values, chunks=7) if execution == "dask" else values
        config = MultiprocConfig(chunks=7) if execution == "mp" else None
        monkeypatch.setattr(reduction.np, "full", bounded_full)
        result, method = reduction._reduce_values(
            [source],
            reduction._normalize_statistics(["median", "nmad"]),
            group_ids=groups,
            total_groups=1_000_000,
            method="map-reduce",
            reindex=False,
            mp_config=config,
        )

        # Check medians and NMAD against NumPy
        assert method == "map-reduce"
        assert result.index.tolist() == [0, 900_000]
        for label in result.index:
            selected = values[groups == label]
            assert result.loc[label, (0, "median")] == np.median(selected)
            assert result.loc[label, (0, "nmad")] == 1.4826 * np.median(np.abs(selected - np.median(selected)))
        assert is_dask_array(source) == (execution == "dask")

    @pytest.mark.parametrize("engine", [None, "numpy", "numba"])
    def test_flox_groupby_reduce__loading_laziness(self, engine: str | None) -> None:
        """Checks that Flox reductions stay lazy until computed."""

        if engine == "numba":
            pytest.importorskip("numba")

        # Create groups spread across uneven chunks
        pytest.importorskip("flox")
        dask = pytest.importorskip("dask")
        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        values = np.arange(2 * 7 * 9, dtype=np.float32).reshape(2, 7, 9)
        groups = np.broadcast_to(np.arange(7)[:, None] % 3, (7, 9))
        expected_groups = [pd.Index([2, 0, 1, 8])]
        expected = _flox_groupby_reduce(
            values,
            [groups],
            expected_groups,
            function="sum",
            use_dask=False,
            fill_value=0,
            dtype=np.float64,
            engine=engine,
        )
        reads = []

        def read_chunk(block: Any, block_id: Any = None) -> Any:
            """Record actual source chunk evaluations without changing their values."""

            if block.size:
                reads.append(block_id)
            return block

        source = da.from_array(values, chunks=(2, 3, 4)).map_blocks(read_chunk, dtype=np.float32)
        lazy_groups = da.from_array(groups, chunks=(3, 4))
        result = _flox_groupby_reduce(
            source,
            [lazy_groups],
            expected_groups,
            function="sum",
            use_dask=True,
            fill_value=0,
            dtype=np.float64,
            engine=engine,
        )
        assert isinstance(source, da.Array)
        assert isinstance(result, da.Array)
        assert not reads

        # Fail if a task assembles every source value before reducing its groups
        def check_task(key: Any, value: Any, *args: Any) -> None:
            if isinstance(value, np.ndarray):
                assert value.shape != values.shape

        # Check equality and one read per chunk
        with dask.config.set(scheduler="synchronous"), Callback(posttask=check_task):
            computed = result.compute()
        assert len(reads) == 9
        np.testing.assert_array_equal(computed, expected)
        assert computed.shape == (2, 4)
        assert isinstance(source, da.Array)

    @pytest.mark.parametrize("reindex", [False, True])
    def test_grouped_quantiles__serialization_and_empty_groups(self, reindex: bool) -> None:
        """Checks that lazy t-digest reductions survive serialization."""

        # Create constant groups for exact quantiles
        pytest.importorskip("crick")
        pytest.importorskip("flox")
        da = pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        groups = np.arange(47) % 3
        values = (groups + 1).astype(float)
        values[:7] = np.nan
        lazy_values = da.from_array(values, chunks=7)
        lazy_groups = da.from_array(groups, chunks=7)
        options: dict[str, Any] = {
            "function": _tdigest_aggregation(100),
            "fill_value": np.nan,
            "dtype": object,
            "sort": True,
            "reindex": reindex,
        }

        # Calculate in-memory medians
        eager_states = _flox_groupby_reduce(values, [groups], [np.arange(5)], use_dask=False, **options)
        expected = _digest_quantile(eager_states, q=0.5)
        completed = []

        # Record evaluated tasks and fail if one assembles all source values
        def check_task(key: Any, value: Any, *args: Any) -> None:
            completed.append(key)
            if isinstance(value, np.ndarray):
                assert value.shape != values.shape

        # Build the lazy median calculation
        with Callback(posttask=check_task):
            states = _flox_groupby_reduce(lazy_values, [lazy_groups], [np.arange(5)], use_dask=True, **options)
            result = states.map_blocks(_digest_quantile, q=0.5, dtype=np.float64)

        # Compute and check equality
        assert isinstance(result, da.Array)
        assert isinstance(lazy_values, da.Array)
        assert not completed
        restored = pickle.loads(pickle.dumps(result))
        with Callback(posttask=check_task):
            computed = restored.compute()
        np.testing.assert_array_equal(computed, expected)
        assert completed
        assert result.shape == (5,)


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

    def test_stats__warn_unknown_global_statistic(self) -> None:
        """Checks that an unknown global statistic warns and leaves known results available."""

        # Three finite values have a mean of two, independent of the unknown request
        values = np.array([1.0, 2.0, 3.0])

        # Global statistics report an unknown name as undefined after warning
        with pytest.warns(UserWarning, match="Statistic name made_up is not recognized"):
            result = gu.stats.stats(values, ["mean", "made_up"])

        # The recognized mean is still calculated
        assert result["mean"] == pytest.approx(2)
        assert np.isnan(result["made_up"])

    def test_resolve_reduction_method__error_invalid(self) -> None:
        """Checks an error is raised for invalid reduction methods."""

        # Check an unknown method
        with pytest.raises(ValueError, match="Argument ``method``"):
            _resolve_reduction_method(["mean"], 2, method="topk")

        # Check custom functions require groupwise
        with pytest.raises(ValueError, match="require method='groupwise'"):
            _resolve_reduction_method([None], 2, method="map-reduce", reindex=True)
        with pytest.raises(ValueError, match="require method='groupwise'"):
            _resolve_reduction_method([None], 2, method="map-reduce", reindex=False)

    @pytest.mark.parametrize("nodata", ["empty", "nan", "masked"])
    def test_reduce_values__empty_sums(self, nodata: str) -> None:
        """Checks that in-memory sums warn and stay undefined when no valid values remain."""

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

    @pytest.mark.parametrize(
        "options,message",
        [
            ({"function": "arbitrary"}, "does not support reduction"),
            ({"function": "nanmedian", "use_dask": True}, "exact median"),
            ({"engine": "unknown"}, "flox_engine"),
            ({"group_values": []}, "matching grouping arrays"),
            ({"expected_groups": []}, "matching grouping arrays"),
        ],
    )
    def test_flox_groupby_reduce__error_unsupported_request(self, options: dict[str, Any], message: str) -> None:
        """Checks an error is raised for unsupported requests before importing Flox."""

        # Create valid reduction inputs and override the unsupported option
        arguments: dict[str, Any] = {
            "values": np.arange(5.0),
            "group_values": [np.zeros(5, dtype=np.int64)],
            "expected_groups": [pd.Index([0])],
            "function": "sum",
            "use_dask": False,
            "fill_value": 0,
            "dtype": np.float64,
            **options,
        }

        # Check validation before the optional backend is imported
        with pytest.raises(ValueError, match=message):
            _flox_groupby_reduce(**arguments)
