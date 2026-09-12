"""Tests 'stratified' subsampling, i.e. subsampling independent within integer group IDs."""

from __future__ import annotations

import warnings
from typing import Literal
from unittest.mock import patch

import numpy as np
import pytest

from geoutils._misc import import_optional
from geoutils.multiproc import MultiprocConfig
from geoutils.sampling.stratified import _stratified_subsample_indices
from geoutils.sampling.subsampling import _splitmix64


class TestStratifiedSubsample:
    """
    Test module for stratified sampling (i.e. independent subsampling within integer groups).

    Tests covering Dask/Multiprocessing backends are located further below in TestStratifiedSubsampleChunked.
    Tests using the parent function stats(by=, subsample_per_group=True) which uses stratified sampling are in
    test_grouping.py, here we test the subfunctions instead. Warning behavior and invalid sample sizes are covered in
    TestStratifiedSubsampleErrors.

    For eager arrays, we check that:
    - Each group receives the requested number or fraction of samples, while negative group IDs are properly excluded.
    - A fixed seed selects the same samples when group IDs or the array shape change.
    """

    def test_stratified_subsample_indices__sample_size_per_group(self) -> None:
        """Checks that a fixed sample size is applied separately to each valid group ID."""

        # Select two positions from each valid group and exclude locations marked with an ID of -1
        groups = np.array([0, 0, 0, -1, 1, 1, 1, 1])
        selected = _stratified_subsample_indices(groups, 2, random_state=42)

        # Check the basic output shape and selected group counts
        assert selected.shape == (4,)
        assert np.array_equal(np.bincount(groups[selected]), [2, 2])
        assert np.all(groups[selected] >= 0)

    @pytest.mark.parametrize("subsample", [1, 2, 5, 100, 0.1, 0.5])
    def test_stratified_subsample_indices__topk_reference(self, subsample: int | float) -> None:
        """Checks that topk selects positions with the lowest seeded random scores within each group."""

        # Use unequal groups and excluded locations so fractions, maximum counts and empty samples are distinguishable
        groups = np.array([0, -1, 0, 1, 1, -1, 1, 1, 1, 1, 1, 1, 2, -1, 2])
        selected = _stratified_subsample_indices(groups, subsample, random_state=42)

        # Build a reference for each group by sorting seeded random scores from the original positions
        expected = []
        for group in (0, 1, 2):
            positions = np.flatnonzero(groups == group)
            sample_size = int(subsample * len(positions)) if subsample <= 1 else min(int(subsample), len(positions))
            keys = _splitmix64(np.uint64(42) ^ positions.astype(np.uint64))
            expected.extend(positions[np.argsort(keys)[:sample_size]])

        # Check the documented output order and confirm that no location is duplicated or selected from ID -1
        expected_indices = np.asarray(expected, dtype=np.int64)
        if subsample == 1:
            expected_indices.sort()
        else:
            keys = _splitmix64(np.uint64(42) ^ expected_indices.astype(np.uint64))
            expected_indices = expected_indices[np.argsort(keys)]
        assert np.array_equal(selected, expected_indices)
        assert len(np.unique(selected)) == len(selected)
        assert np.all(groups[selected] >= 0)

    def test_stratified_subsample_indices__original_positions(self) -> None:
        """
        Checks that "topk" strategy uses original positions across group IDs, array shapes for a given random seed.
        """

        # Assign different positive IDs to the same groups and reshape their locations
        groups = np.repeat([0, 1, 2], [2, 8, 5])
        expected = _stratified_subsample_indices(groups, 3, random_state=42)
        remapped = _stratified_subsample_indices(100 - groups, 3, random_state=42)
        reshaped = _stratified_subsample_indices(groups.reshape(3, 5), 3, random_state=42)
        assert np.array_equal(remapped, expected)
        assert np.array_equal(reshaped, expected)

        # Check that one seed is drawn from a supplied random generator for the complete sampling call
        actual_rng = np.random.default_rng(9)
        reference_rng = np.random.default_rng(9)
        seed = int(reference_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        selected = _stratified_subsample_indices(groups, 3, random_state=actual_rng)
        assert np.array_equal(selected, _stratified_subsample_indices(groups, 3, random_state=seed))
        assert actual_rng.integers(100000) == reference_rng.integers(100000)


class TestStratifiedSubsampleChunked:
    """Checks stratified subsampling against eager results for Dask arrays and Multiproc chunks."""

    @pytest.mark.parametrize(
        "shape,chunks", [((120,), (7,)), ((10, 12), (3, 5)), ((10, 12), (4, 4)), ((10, 12), (12, 12))]
    )
    @pytest.mark.parametrize("subsample", [1, 3, 0.15, 0.5])
    @pytest.mark.parametrize("strategy", ["topk", "sequential"])
    def test_stratified_subsample_indices__eager_dask_and_multiproc(
        self,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        subsample: int | float,
        strategy: Literal["topk", "sequential"],
    ) -> None:
        """
        Checks that stratified subsampling selects exactly the same positions (for "topk" strategy) and behave the
        same depending on chunks (for "sequential" strategy).
        """

        # Create repeating group IDs, exclude every seventh location, and add one group containing a single location
        import_optional("dask")
        import dask.array as da

        groups = (np.arange(120) % 4).reshape(shape)
        groups.ravel()[::7] = -1
        groups.ravel()[1] = 4
        lazy = da.from_array(groups, chunks=chunks)
        config = MultiprocConfig(chunks=chunks[0] if len(chunks) == 1 else (chunks[0], chunks[1]))

        # Sample the same group IDs through the eager, Dask and Multiproc backends
        eager = _stratified_subsample_indices(groups, subsample, random_state=42, strategy=strategy)
        dask_result = _stratified_subsample_indices(lazy, subsample, random_state=42, strategy=strategy)
        tile_result = _stratified_subsample_indices(
            groups, subsample, random_state=42, strategy=strategy, mp_config=config
        )

        # Inputs remain chunked, while this index-selection helper returns eager NumPy positions
        assert isinstance(lazy, da.Array)
        assert isinstance(dask_result, np.ndarray) and isinstance(tile_result, np.ndarray)

        # Fractions must be rounded from the full group sizes, rather than once in each chunk
        _, counts = np.unique(groups[groups >= 0], return_counts=True)
        expected_counts = (counts * subsample).astype(int) if subsample <= 1 else np.minimum(counts, int(subsample))
        for selected in (eager, dask_result, tile_result):
            assert np.array_equal(np.bincount(groups.ravel()[selected], minlength=5), expected_counts)
            assert len(np.unique(selected)) == len(selected)
        assert np.array_equal(dask_result, tile_result)

        # Topk is independent of chunks; sequential sampling repeats when the chunk layout stays the same
        if strategy == "topk":
            assert np.array_equal(dask_result, eager)
        else:
            repeated = _stratified_subsample_indices(lazy, subsample, random_state=42, strategy=strategy)
            assert np.array_equal(dask_result, repeated)

    @pytest.mark.parametrize("strategy", ["topk", "sequential"])
    def test_stratified_subsample_indices__multiproc_workers(
        self,
        strategy: Literal["topk", "sequential"],
    ) -> None:
        """Checks that Multiproc workers combine several task batches and match the positions selected with Dask."""

        # Use more than eight tiles so Multiproc processes several task batches and combines their samples
        from geoutils.multiproc.cluster import MpCluster

        import_optional("dask")
        import dask.array as da

        groups = (np.arange(360) % 7).reshape(18, 20)
        groups[::3, ::4] = -1
        lazy = da.from_array(groups, chunks=(3, 4))
        expected = _stratified_subsample_indices(lazy, 0.3, 42, strategy)
        eager = _stratified_subsample_indices(groups, 0.3, 42, strategy)

        # Run the same tile layout through two Multiproc worker processes and compare the selected positions
        with MpCluster({"nb_workers": 2, "max_tasks_per_child": None}) as cluster:
            config = MultiprocConfig(chunks=(3, 4), cluster=cluster)
            selected = _stratified_subsample_indices(groups, 0.3, 42, strategy, mp_config=config)
        assert np.array_equal(selected, expected)
        assert isinstance(lazy, da.Array) and isinstance(selected, np.ndarray)
        if strategy == "topk":
            assert np.array_equal(selected, eager)

    @pytest.mark.parametrize("strategy", ["topk", "sequential"])
    def test_stratified_subsample_indices__empty_groups(self, strategy: Literal["topk", "sequential"]) -> None:
        """Checks that ID -1 and sample sizes rounded to zero return no positions for eager and Dask inputs."""

        # Create one input with no valid group ID and another with one valid location in its final Dask block
        import_optional("dask")
        import dask.array as da

        all_excluded = np.full((4, 6), -1)
        tiny_group = all_excluded.copy()
        tiny_group[-1, -1] = 0

        # 50% of a one-member group rounds to zero (same floor rule that we use for ordinary subsampling)
        for groups in (all_excluded, tiny_group):
            expected = _stratified_subsample_indices(groups, 0.5, random_state=42, strategy=strategy)
            lazy = da.from_array(groups, chunks=(2, 3))
            selected = _stratified_subsample_indices(lazy, 0.5, random_state=42, strategy=strategy)
            assert isinstance(lazy, da.Array) and isinstance(selected, np.ndarray)
            assert np.array_equal(selected, expected)
            assert selected.size == 0
            assert np.issubdtype(selected.dtype, np.integer)

        # Check that a sample size of two selects the only valid location despite the other empty Dask blocks
        expected = _stratified_subsample_indices(tiny_group, 2, random_state=42, strategy=strategy)
        lazy = da.from_array(tiny_group, chunks=(2, 3))
        selected = _stratified_subsample_indices(lazy, 2, random_state=42, strategy=strategy)
        assert np.array_equal(selected, expected)
        assert np.array_equal(selected, [23])


class TestStratifiedSubsampleErrors:
    """Test module for validation errors and warning behavior in stratified subsampling."""

    def test_stratified_subsample_indices__per_group_selection(self) -> None:
        """
        Checks that partial selection sorts groups and includes every position from smaller groups without warnings.
        """

        # Use groups of very different sizes so each partial sort reveals exactly which group slice it received
        groups = np.repeat([10, 20, 30], [2, 20, 200])
        with warnings.catch_warnings(record=True) as recorded:
            with patch("numpy.argpartition", wraps=np.argpartition) as partition:
                selected = _stratified_subsample_indices(groups, 5, random_state=42)

        # The two larger groups are partially selected; the first group already fits within its sample size
        assert [call.args[0].size for call in partition.call_args_list] == [20, 200]
        assert np.array_equal(np.unique(groups[selected], return_counts=True)[1], [2, 5, 5])
        assert not recorded

    @pytest.mark.parametrize("subsample", [0, -1, np.inf, np.nan])
    def test_stratified_subsample_indices__error_invalid_subsample(self, subsample: int | float) -> None:
        """Checks that invalid sampling amounts fail even when all locations are excluded."""

        with pytest.raises(ValueError, match="positive finite"):
            _stratified_subsample_indices(np.full(5, -1), subsample)
