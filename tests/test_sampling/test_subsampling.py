"""Test array, point cloud, and raster subsampling."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from numpy.typing import NDArray
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils import open_raster
from geoutils._dispatch import is_dask_dataframe
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.raster.array import get_mask_from_array
from geoutils.sampling.subsampling import (
    _recover_splitmix64_indices,
    _sample_valid_indices,
    _splitmix64,
)
from geoutils.sampling.subsampling import _subsample as _subsample_values
from geoutils.sampling.subsampling import (
    _subsample_numpy,
)


class TestArraySubsample:
    """
    Test module for eager subsampling with one-, two-, and three-dimensional masked arrays.

    Tests specific to point/raster inputs are available in other modules below, as well as for the testing across
    Dask and Multiprocessing backends!
    """

    # One-dimensional array with one masked value
    array1D = np.ma.masked_array(np.arange(10), mask=np.zeros(10))
    array1D.mask[3] = True
    assert np.ndim(array1D) == 1
    assert np.count_nonzero(array1D.mask) > 0

    # Two-dimensional array with one masked value
    array2D = np.ma.masked_array(np.arange(9).reshape((3, 3)), mask=np.zeros((3, 3)))
    array2D.mask[0, 1] = True
    assert np.ndim(array2D) == 2
    assert np.count_nonzero(array2D.mask) > 0

    # Three-dimensional array with one masked value
    array3D = np.ma.masked_array(np.arange(9).reshape((1, 3, 3)), mask=np.zeros((1, 3, 3)))
    array3D = np.ma.vstack((array3D, array3D + 10))
    array3D.mask[0, 0, 1] = True
    assert np.ndim(array3D) == 3
    assert np.count_nonzero(array3D.mask) > 0

    @pytest.mark.parametrize(
        "array",
        [
            np.arange(10),
            np.arange(9).reshape((3, 3)),
            np.arange(18).reshape((2, 3, 3)),
            array1D,
            array2D,
            array3D,
        ],
    )
    def test_subsample__inputs(self, array: NDArrayNum) -> None:
        """Checks that counts, fractions, returned indexes, and random seeds follow the public sampling rules."""

        warnings.filterwarnings("ignore", message=".*larger than the number of valid pixels.*", category=UserWarning)
        input_mask = np.ma.getmaskarray(array)

        # Check every requested count below the input size
        for npts in np.arange(2, np.size(array)):
            random_values = _subsample_numpy(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(np.ma.getmaskarray(random_values)) == 0

        # Check that a count above the available values returns every available value
        random_values = _subsample_numpy(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~input_mask])

        # Check that 1 returns every available value in the original order
        random_values = _subsample_numpy(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~input_mask])

        random_values_2 = _subsample_numpy(array, subsample=1)
        assert np.array_equal(random_values, random_values_2)

        # Check that a fraction between 0 and 1 returns the right amount of valid values
        random_values = _subsample_numpy(array, subsample=0.5)
        assert np.size(random_values) == int(np.count_nonzero(~input_mask) * 0.5)

        # Check returned indexes against the input dimensions and requested fraction
        indices = _subsample_numpy(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.count_nonzero(~input_mask) * 0.3)

        # Check that top-k is the default strategy
        sub42 = _subsample_numpy(array, subsample=10, random_state=42)
        topk42 = _subsample_numpy(array, subsample=10, random_state=42, strategy="topk")
        assert np.array_equal(sub42, topk42)

        # Check that sequential sampling accepts an integer seed or the matching NumPy generator
        sub42 = _subsample_numpy(array, subsample=10, random_state=42, strategy="sequential")
        rng = np.random.default_rng(42)
        sub42_gen = _subsample_numpy(array, subsample=10, random_state=rng, strategy="sequential")
        assert np.array_equal(sub42, sub42_gen)

    def test_splitmix64__recover_indices(self) -> None:
        """
        Checks that random keys can recover their global cell indexes (used for performance, to avoid carrying
        values along the keys).
        """

        # Create indexes spanning the signed range used by NumPy array positions
        seed = 42
        expected = np.array([0, 1, 2, 2**31, 2**48 + 17, 2**62 - 1], dtype=np.int64)
        keys = np.asarray(_splitmix64(np.uint64(seed) ^ expected.astype(np.uint64)), dtype=np.uint64)

        # Reverse the keys and compare to original indexes
        recovered = _recover_splitmix64_indices(keys, seed)
        assert np.shares_memory(keys, recovered)
        np.testing.assert_array_equal(recovered, expected)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestArraySubsampleChunked:
    """Test module for subsample() across eager, Dask and multiprocessing inputs.

    We check that backends return the requested sample size and exactly the same values (for "topk" strategy).
    We also check behaviour with selected bands, mask input and chunk size.
    """

    # Strategies supported by _subsample()
    subsample_strategies = ("sequential", "topk")

    @pytest.mark.parametrize("path_index", [0, 2])
    @pytest.mark.parametrize("strategy", subsample_strategies)
    @pytest.mark.parametrize("return_indices", [False, True])
    @pytest.mark.parametrize("subsample", [2, 100, 0.05])  # int size and fraction
    def test_subsample__backends(
        self,
        path_index: int,
        strategy: Literal["sequential", "topk"],
        return_indices: bool,
        subsample: int | float,
        lazy_test_files_tiny: list[str],
    ) -> None:
        """Checks that all storage paths follow the same sampling, loading, and repeatability rules."""

        import dask.array as da

        warnings.filterwarnings("ignore", category=UserWarning, message="Argument ``subsample`` with value*")

        # 1/ Open matching inputs for NumPy, Dask, and multiprocessing
        path_raster = lazy_test_files_tiny[path_index]

        # The two NumPy calls use loaded Raster and Xarray inputs
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        assert raster_base.is_loaded

        ds_base = open_raster(path_raster)
        ds_base.load()
        assert ds_base._in_memory

        # Worker processes read tiles from a Raster that stays linked to its file
        raster_mp = gu.Raster(path_raster)
        assert not raster_mp.is_loaded

        # The Xarray accessor keeps the chunked Dask input lazy
        ds_dask = open_raster(path_raster, chunks={"x": 10, "y": 10})
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert ds_dask.data.chunks is not None

        # 2/ Run every storage path with the same seed
        seed = 42
        mp_config = MultiprocConfig(chunks=(10, 7))

        # NumPy through Raster
        out_raster = _subsample_values(
            raster_base,
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # NumPy through an Xarray accessor
        out_xr = _subsample_values(
            ds_base.rst,
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # Dask through an Xarray accessor
        out_dask = _subsample_values(
            ds_dask.rst,
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # Multiproc workers through Raster
        out_mp = _subsample_values(
            raster_mp,
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
            mp_config=mp_config,
        )

        # 3/ Check that Dask and file-backed inputs stay unloaded
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)

        assert not raster_mp.is_loaded

        # Dask outputs also stay lazy until values are compared
        if return_indices:
            assert isinstance(out_dask, tuple) and len(out_dask) == 2
            assert isinstance(out_dask[0], da.Array)
            assert isinstance(out_dask[1], da.Array)
        else:
            assert isinstance(out_dask, da.Array)

        # 4/ Load each small result as NumPy arrays for comparison

        def _as_numpy(
            out: object,
        ) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
            """Convert any returned values or indexes to NumPy arrays."""
            if isinstance(out, tuple):
                r, c = out
                if hasattr(r, "compute"):
                    r = r.compute()
                if hasattr(c, "compute"):
                    c = c.compute()
                return (np.asarray(r), np.asarray(c))
            else:
                if hasattr(out, "compute"):
                    out = out.compute()
                return np.asarray(out)

        out_raster_np = _as_numpy(out_raster)
        out_xr_np = _as_numpy(out_xr)
        out_dask_np = _as_numpy(out_dask)
        out_mp_np = _as_numpy(out_mp)

        # 5/ Check the result size and values shared by every storage path
        # _subsample() uses band one by default
        arr = raster_base.data if raster_base.data.ndim == 2 else raster_base.data[0, :, :]
        assert arr.ndim == 2

        mask = get_mask_from_array(arr)
        n_valid = int(np.count_nonzero(~mask))

        # Calculate the expected sample size independently from the implementation helper
        if isinstance(subsample, float):
            expected = int(subsample * n_valid)
        else:
            expected = min(int(subsample), n_valid)

        def _check_output(out_np: NDArrayNum | tuple[NDArrayNum, NDArrayNum]) -> None:
            """Check one output's size and that every returned position is available."""
            if isinstance(out_np, tuple):
                rr, cc = out_np
                assert rr.shape == cc.shape
                assert rr.ndim == 1 and cc.ndim == 1
                assert len(rr) == expected
                # Returned rows and columns must lie inside the raster
                assert np.all((0 <= rr) & (rr < arr.shape[0]))
                assert np.all((0 <= cc) & (cc < arr.shape[1]))
                # Every returned pixel must be available according to the shared raster mask
                assert np.all(~mask[rr, cc])
            else:
                assert out_np.ndim == 1
                assert len(out_np) == expected
                assert np.all(np.isfinite(out_np))

        _check_output(out_raster_np)
        _check_output(out_xr_np)
        _check_output(out_dask_np)
        _check_output(out_mp_np)

        # 6/ Check exact agreement for `topk` and repeatability for `sequential`
        if strategy == "topk":
            assert np.array_equal(out_raster_np, out_xr_np)
            assert np.array_equal(out_raster_np, out_dask_np)
            assert np.array_equal(out_raster_np, out_mp_np)
        else:
            # Sequential sampling depends on chunk order, so repeat each path with the same seed
            out_raster_np_2 = _as_numpy(
                _subsample_values(
                    raster_base,
                    subsample=subsample,
                    return_indices=return_indices,
                    random_state=seed,
                    strategy=strategy,
                )
            )
            out_dask_np_2 = _as_numpy(
                _subsample_values(
                    ds_dask.rst,
                    subsample=subsample,
                    return_indices=return_indices,
                    random_state=seed,
                    strategy=strategy,
                )
            )
            out_mp_np_2 = _as_numpy(
                _subsample_values(
                    raster_mp,
                    subsample=subsample,
                    return_indices=return_indices,
                    random_state=seed,
                    strategy=strategy,
                    mp_config=mp_config,
                )
            )

            assert np.array_equal(out_raster_np, out_raster_np_2)
            assert np.array_equal(out_dask_np, out_dask_np_2)
            assert np.array_equal(out_mp_np, out_mp_np_2)

        # 7/ Check that returned indexes select the same values as value mode
        if return_indices:
            rr, cc = out_raster_np
            vals_from_indices = arr[rr, cc]
            vals_raster = _subsample_values(
                raster_base,
                subsample=subsample,
                return_indices=False,
                random_state=seed,
                strategy=strategy,
            )
            vals_raster_np = _as_numpy(vals_raster)
            assert np.array_equal(np.asarray(vals_from_indices), np.asarray(vals_raster_np))

    @pytest.mark.parametrize("strategy", subsample_strategies)
    @pytest.mark.parametrize("mask_form", ["array", "masked", "raster", "vector"])
    def test_subsample__masked_backends(
        self,
        strategy: Literal["sequential", "topk"],
        mask_form: str,
        lazy_test_files_tiny: list[str],
        tmp_path: Path,
    ) -> None:
        """Checks that masks and finite data jointly define fractional samples across eager, Dask and worker tiles."""

        import dask.array as da

        from geoutils.multiproc.cluster import MpCluster

        # Prepare a common mask and compute the eligible population from the original raster values
        path = lazy_test_files_tiny[0]
        raster = gu.Raster(path, load_data=True)
        values = raster.data
        allowed = np.indices(raster.shape)[1] < raster.width // 2
        mask: Any = allowed.copy()

        # Use equivalent spatial selections, with extra nodata entries for the masked-array case
        if mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[::3, ::4] = True
            allowed &= ~mask.mask
        elif mask_form == "raster":
            mask = gu.Raster.from_array(mask, raster.transform, raster.crs)
        elif mask_form == "vector":
            left, bottom, _, top = raster.bounds
            middle = left + (raster.width // 2) * raster.res[0]
            mask = gpd.GeoDataFrame(geometry=[box(left, bottom, middle, top)], crs=raster.crs)

        # Apply the fraction after excluding source nodata values and mask entries that are nodata or False
        eligible = allowed & ~get_mask_from_array(values)
        expected_size = int(eligible.sum() * 0.25)
        assert expected_size > 0

        # Sample with different chunk layouts and check exact source grid positions while inputs remain lazy
        lazy = open_raster(path, chunks={"x": 9, "y": 7})
        unloaded = gu.Raster(path)
        lazy_mask = da.from_array(mask, chunks=(5, 8)) if mask_form == "masked" else mask
        worker_mask = mask

        # Leave the boolean mask file unloaded so workers also read its pixels by tile
        if mask_form == "raster":
            mask_path = tmp_path / "sampling_mask.tif"
            mask.to_file(mask_path)
            worker_mask = gu.Raster(mask_path, is_mask=True)
        results = []

        # Real processes check that both count and selection workers receive the same sliced mask
        with MpCluster({"nb_workers": 2, "max_tasks_per_child": None}) as cluster:
            config = MultiprocConfig(chunks=(6, 10), cluster=cluster)
            sources = [(raster, mask, None), (lazy.rst, lazy_mask, None), (unloaded, worker_mask, config)]
            for source, source_mask, source_config in sources:
                options: dict[str, Any] = {
                    "mask": source_mask,
                    "random_state": 42,
                    "strategy": strategy,
                    "mp_config": source_config,
                }
                indices = _subsample_values(source, 0.25, return_indices=True, **options)
                sampled = _subsample_values(source, 0.25, **options)

                # Compute only the small sample results; the original raster objects are not loaded
                indices = tuple(
                    np.asarray(index.compute() if hasattr(index, "compute") else index) for index in indices
                )
                sampled = sampled.compute() if hasattr(sampled, "compute") else sampled
                assert len(indices) == 2 and len(indices[0]) == expected_size
                assert np.all(eligible[indices])
                assert len(np.unique(np.ravel_multi_index(indices, raster.shape))) == expected_size
                np.testing.assert_array_equal(sampled, values[indices])
                assert sampled.dtype == values.dtype
                results.append(indices)

        # Check that inputs are not loaded and topk returns the same sample for every chunk layout
        assert not lazy._in_memory and isinstance(lazy.data, da.Array)
        assert not unloaded.is_loaded
        if mask_form == "raster":
            assert not worker_mask.is_loaded
        if strategy == "topk":
            np.testing.assert_array_equal(results[0], results[1])
            np.testing.assert_array_equal(results[0], results[2])

    @pytest.mark.parametrize("subsample", [1, 10])
    def test_subsample__keep_nodata_backends(self, subsample: int, tmp_path: Path) -> None:
        """Checks that subsamples nodata cells are processed in the same order across every backend."""

        import dask.array as da

        # 1/ Write a mostly nodata raster with small chunksize (that could trigger row-order mistakes if nodata cells
        # were managed inconsistently)
        values = np.ma.array(np.arange(24, dtype=np.int16).reshape(4, 6), mask=True)
        values.mask[0, 0] = False
        values.mask[3, 4] = False
        raster = gu.Raster.from_array(values, from_origin(0, 4, 1, 1), crs=32633, nodata=-9999)
        path = tmp_path / "keep_nodata_sample.tif"
        raster.to_file(path)

        # 2/ Keep the first five columns with the mask
        allowed = np.indices(raster.shape)[1] < 5
        options: dict[str, Any] = {
            "subsample": subsample,
            "return_indices": True,
            "random_state": 42,
            "strategy": "topk",
            "skip_nodata": False,
            "mask": allowed,
        }
        expected = _subsample_values(raster, **options)
        lazy = open_raster(str(path), chunks={"x": 4, "y": 3})
        unloaded = gu.Raster(path)

        # 3/ Select cells lazily through 3 x 4 chunk without loading source
        lazy_result = _subsample_values(lazy.rst, **options)
        worker_result = _subsample_values(
            unloaded,
            **options,
            mp_config=MultiprocConfig(chunks=(3, 4)),
        )
        assert all(isinstance(index, da.Array) for index in lazy_result)
        assert not lazy._in_memory and not unloaded.is_loaded
        lazy_result = tuple(index.compute() for index in lazy_result)

        # 4/ Check the exact eager order and confirm that nodata cells are treated similarly
        np.testing.assert_array_equal(lazy_result, expected)
        np.testing.assert_array_equal(worker_result, expected)
        assert len(expected[0]) == (allowed.sum() if subsample == 1 else subsample)
        assert np.all(allowed[expected])
        assert np.any(values.mask[expected])
        if subsample == 1:
            expected_flat = np.flatnonzero(allowed)
            np.testing.assert_array_equal(np.ravel_multi_index(expected, raster.shape), expected_flat)

    def test_subsample__topk_matches_dask_argtopk(self, tmp_path: Path) -> None:
        """Checks that Dask and multiprocessing top-k match Dask argtopk for identical cell keys."""

        import dask.array as da

        # Create uneven chunks with nodata cells in every part of the flattened raster
        values = np.arange(99, dtype=np.float32).reshape((9, 11))
        values.ravel()[::13] = np.nan
        lazy_values = da.from_array(values, chunks=(4, 5))
        source = gu.RasterAccessor.from_array(lazy_values, from_origin(0, 9, 1, 1), 32633)
        source_file = tmp_path / "topk-reference.tif"
        gu.Raster.from_array(values, from_origin(0, 9, 1, 1), 32633, nodata=-99999).to_file(source_file)
        unloaded = gu.Raster(source_file)
        seed = 42
        sample_size = 17

        # Select cells through both GeoUtils chunked paths while keeping the Dask indexes lazy
        rows, columns = _subsample_values(
            source.rst,
            sample_size,
            return_indices=True,
            random_state=seed,
            strategy="topk",
        )
        worker_rows, worker_columns = _subsample_values(
            unloaded,
            sample_size,
            return_indices=True,
            random_state=seed,
            strategy="topk",
            mp_config=MultiprocConfig(chunks=(4, 5)),
        )
        assert isinstance(rows, da.Array) and isinstance(columns, da.Array)

        # Give every eligible flat cell the same deterministic key used by GeoUtils, then use Dask's reduction
        valid = da.isfinite(lazy_values.reshape(-1))
        cell_numbers = da.arange(values.size, chunks=valid.chunks, dtype=np.int64)
        key_input = np.uint64(seed) ^ cell_numbers.astype(np.uint64)
        keys = key_input.map_blocks(_splitmix64, dtype=np.uint64)
        eligible_keys = da.where(valid, keys, np.iinfo(np.uint64).max)
        expected = da.argtopk(eligible_keys, -sample_size, split_every=2).compute()

        # Compare complete cell numbers and confirm that neither source was loaded
        actual = (rows * values.shape[1] + columns).compute()
        worker_actual = worker_rows * values.shape[1] + worker_columns
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(worker_actual, expected)
        assert not source._in_memory and not unloaded.is_loaded

    @pytest.mark.parametrize("strategy", ["sequential", "topk"])
    @pytest.mark.parametrize("shape", [(30,), (5, 6)])
    def test_sample_valid_indices__boolean_mask(
        self, strategy: Literal["sequential", "topk"], shape: tuple[int, ...]
    ) -> None:
        """Checks that boolean False locations are excluded and topk preserves positions across array chunks."""

        # Make both accepted and excluded positions occur in every chunk
        import dask.array as da

        valid = (np.arange(30) % 4 != 0).reshape(shape)
        lazy = da.from_array(valid, chunks=3)
        eager = _sample_valid_indices(valid, subsample=5, random_state=42, strategy=strategy)
        result = _sample_valid_indices(lazy, subsample=5, random_state=42, strategy=strategy)

        # Sequential grid traversal can differ across layouts; point rows and topk have identical positions
        assert isinstance(lazy, da.Array)
        assert all(isinstance(index, np.ndarray) for index in result)
        assert len(result) == len(shape)
        assert len(result[0]) == 5
        assert np.all(valid[result])
        assert len(np.unique(np.ravel_multi_index(result, shape))) == 5
        if strategy == "topk" or len(shape) == 1:
            np.testing.assert_array_equal(result, eager)

    @pytest.mark.parametrize("opened_bands,selected_band", [(None, 2), ([2], 1)])
    def test_subsample__multiproc_selected_band(
        self, opened_bands: list[int] | None, selected_band: int, tmp_path: Path
    ) -> None:
        """Checks that masked multiprocessing samples the requested disk band without changing the source band list."""

        # Write distinct integer bands with one nodata value in the second band's eligible half
        values = np.arange(24, dtype=np.int16).reshape(4, 6)
        data = np.ma.array(np.stack([values + 100, values]), mask=False)
        data.mask[1, 1, 4] = True
        raster = gu.Raster.from_array(data, from_origin(0, 4, 1, 1), crs=32633, nodata=-9999)
        path = tmp_path / "multiband_sample.tif"
        raster.to_file(path)

        # Selecting an already restricted Raster counts from its available bands, not the original disk bands
        source = gu.Raster(path, bands=opened_bands)
        original_bands = source.bands
        mask = np.indices(source.shape)[1] >= 3
        eligible = mask & ~data.mask[1]
        config = MultiprocConfig(chunks=(2, 3))

        # Select all eligible pixels so the expected population is independent of random draws and worker tile order
        indices = _subsample_values(source, 1, band=selected_band, return_indices=True, mask=mask, mp_config=config)
        sampled = _subsample_values(source, 1, band=selected_band, mask=mask, mp_config=config)
        assert len(indices[0]) == eligible.sum()
        assert np.all(eligible[indices])
        np.testing.assert_array_equal(sampled, values[indices])
        assert sampled.dtype == values.dtype
        assert source.bands == original_bands and not source.is_loaded

    @pytest.mark.parametrize("return_indices", [False, True])
    def test_subsample__multiproc_large_sample_npy(
        self,
        return_indices: bool,
        lazy_test_files_tiny: list[str],
        tmp_path: Path,
    ) -> None:
        """Checks that a large Multiproc sample is returned as a NumPy file."""

        # Use a sample larger than every 10 x 7 tile so selection and output both follow the chunked path
        source = gu.Raster(lazy_test_files_tiny[0])
        eager = gu.Raster(lazy_test_files_tiny[0], load_data=True)
        output_file = tmp_path / ("indices.npy" if return_indices else "values.npy")
        options: dict[str, Any] = {
            "subsample": 100,
            "return_indices": return_indices,
            "random_state": 42,
            "strategy": "topk",
        }
        expected = _subsample_values(eager, **options)

        # Write the sample to a NumPy file
        result = _subsample_values(
            source,
            **options,
            mp_config=MultiprocConfig(chunks=(10, 7), outfile=str(output_file)),
        )
        stored = np.load(output_file, mmap_mode="r")

        # Check the stored shape, type, and values
        expected_array = np.asarray(expected)
        result_array = np.asarray(result)
        expected_shape = (2, 100) if return_indices else (100,)
        assert stored.shape == expected_shape
        assert isinstance(stored, np.memmap)
        if return_indices:
            assert all(isinstance(index, np.memmap) for index in result)
        else:
            assert isinstance(result, np.memmap)
        np.testing.assert_array_equal(stored, expected_array)
        np.testing.assert_array_equal(result_array, expected_array)
        assert not source.is_loaded

    def test_subsample__multiproc_force_output_to_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        lazy_test_files_tiny: list[str],
        tmp_path: Path,
    ) -> None:
        """Checks that forcing memory output bypasses cutoff search and does not create a NumPy file."""

        from geoutils.sampling import subsampling as subsampling_module

        # Reject cutoff search for a sample that would otherwise exceed one 10 x 7 tile
        def reject_cutoff(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("Forced memory output must bypass cutoff search.")

        monkeypatch.setattr(subsampling_module, "_multiproc_array_topk_cutoff", reject_cutoff)
        source = gu.Raster(lazy_test_files_tiny[0])
        output_file = tmp_path / "forced.npy"

        # Collect the ordinary NumPy result even though the requested sample exceeds one worker tile
        result = _subsample_values(
            source,
            subsample=100,
            random_state=42,
            strategy="topk",
            mp_config=MultiprocConfig(chunks=(10, 7), outfile=str(output_file)),
            force_output_to_memory=True,
        )

        # Verify the configured file is untouched and the source raster unloaded
        assert isinstance(result, np.ndarray) and not isinstance(result, np.memmap)
        assert result.shape == (100,)
        assert not output_file.exists()
        assert not source.is_loaded

    @pytest.mark.parametrize("backend", ["numpy", "dask", "multiprocessing"])
    @pytest.mark.parametrize("tiny_fraction", [False, True])
    @pytest.mark.parametrize("strategy", ["sequential", "topk"])
    def test_subsample__raster_empty_masks(
        self, backend: str, tiny_fraction: bool, strategy: Literal["sequential", "topk"], tmp_path: Path
    ) -> None:
        """Checks that empty raster samples have the source value dtype and expected index dimensions."""

        # Select no pixels or one pixel whose half-sample rounds to zero, crossing several worker tiles
        values = np.arange(24, dtype=np.int16).reshape(4, 6)
        raster = gu.Raster.from_array(values, from_origin(0, 4, 1, 1), crs=32633)
        mask = np.zeros(raster.shape, dtype=bool)
        subsample = 1.0
        if tiny_fraction:
            mask[-1, -1] = True
            subsample = 0.5
        expected_sample = _subsample_values(raster, subsample, mask=mask, random_state=42, strategy=strategy)
        expected_indices = _subsample_values(
            raster, subsample, return_indices=True, mask=mask, random_state=42, strategy=strategy
        )

        # Write an unloaded file for multiprocessing and the lazy accessor for Dask
        source: Any = raster
        config = None
        if backend != "numpy":
            path = tmp_path / "empty_sample.tif"
            raster.to_file(path)
            if backend == "dask":
                source = open_raster(str(path), chunks={"x": 3, "y": 2}).rst
            else:
                source = gu.Raster(path)
                config = MultiprocConfig(chunks=(2, 3))

        # Return empty samples and two empty coordinate arrays without changing the source value dtype
        sampled = _subsample_values(source, subsample, mask=mask, random_state=42, strategy=strategy, mp_config=config)
        indices = _subsample_values(
            source, subsample, return_indices=True, mask=mask, random_state=42, strategy=strategy, mp_config=config
        )
        assert sampled.size == 0 and sampled.dtype == np.dtype(source.dtype)
        assert len(indices) == 2 and all(index.size == 0 for index in indices)
        assert all(np.issubdtype(index.dtype, np.integer) for index in indices)
        assert np.array_equal(sampled, expected_sample)
        assert np.array_equal(indices, expected_indices)
        if backend == "dask":
            import dask.array as da

            assert isinstance(source.data, da.Array) and not source._obj._in_memory
        if backend == "multiprocessing":
            assert not source.is_loaded


class TestPointSubsample:
    """
    Test module for subsample() on point values.

    We only check behaviour with input masks on point data here. Other array tests are done directly in
    TestArraySubsample, and chunked point tests are in TestPointSubsampleChunked.
    """

    def test_subsample__point_output_and_array(self) -> None:
        """Checks that point subsampling returns complete point rows by default and values when requested."""

        # Give the source extra attributes so the point output must preserve more than its sampled data values
        values = np.arange(12, dtype=np.int16)
        points = gu.PointCloud.from_xyz(values, np.zeros(12), values, crs=32633)
        points.ds["label"] = [f"point-{index}" for index in range(12)]
        expected_indices = points.subsample(5, return_indices=True, as_array=True, random_state=42)[0]

        # Request the default point output and the explicit one-dimensional array output
        sampled_points = points.subsample(5, random_state=42)
        sampled_dataframe = points.ds.pc.subsample(5, random_state=42)
        sampled_values = points.subsample(5, as_array=True, random_state=42)

        # Check point and array outputs
        assert isinstance(sampled_points, gu.PointCloud)
        assert isinstance(sampled_dataframe, gpd.GeoDataFrame)
        assert sampled_points.ds.equals(points.ds.iloc[expected_indices])
        assert sampled_dataframe.equals(points.ds.iloc[expected_indices])
        np.testing.assert_array_equal(sampled_values, values[expected_indices])

    @pytest.mark.parametrize(
        "mask_form",
        [
            "array",
            "list",
            "tuple",
            "masked",
            "xarray",
            "vector",
            "vector_accessor",
            "raster",
            "raster_accessor",
            "pointcloud",
            "point_accessor",
        ],
    )
    def test_subsample__point_masks(self, mask_form: str) -> None:
        """Checks all types of input inlier masks."""

        # Give the points duplicate labels so only positional indexes can identify the sampled rows
        y, x = np.mgrid[:6, :6]
        values = np.arange(36, dtype=np.int16)
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        points.ds.index = np.arange(36) % 3
        original = points.ds.copy()
        eligible = (x < 3).ravel()
        mask: Any = eligible.copy()

        # Build the same mask with different input types, covering only half of the input
        if mask_form == "list":
            mask = mask.tolist()
        elif mask_form == "tuple":
            mask = tuple(mask.tolist())
        elif mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[7] = True
            eligible[7] = False
        elif mask_form == "xarray":
            mask = xr.DataArray(mask.reshape(6, 6), dims=("row", "column"))
        elif mask_form in {"vector", "vector_accessor"}:
            vector_mask = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 2.5, 5.5)], crs=points.crs)
            mask = gu.Vector(vector_mask) if mask_form == "vector" else vector_mask
        elif mask_form in {"raster", "raster_accessor"}:
            raster_values = (x < 3)[::-1]
            transform = from_origin(0, 5, 1, 1)
            if mask_form == "raster":
                mask = gu.Raster.from_array(raster_values, transform, points.crs)
            else:
                mask = gu.RasterAccessor.from_array(raster_values, transform, points.crs)
        elif mask_form in {"pointcloud", "point_accessor"}:
            point_mask = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), eligible, crs=points.crs)
            mask = point_mask if mask_form == "pointcloud" else point_mask.ds

        # Draw the expected top-k samples for a generator and an integer seed
        eligible_values = np.ma.array(values, mask=~eligible)
        expected_rng = np.random.default_rng(42)
        expected_indices = _subsample_numpy(eligible_values, 0.5, return_indices=True, random_state=expected_rng)[0]
        expected_seed_indices = _subsample_numpy(eligible_values, 0.5, return_indices=True, random_state=42)[0]
        actual_rng = np.random.default_rng(42)
        indices = points.subsample(0.5, return_indices=True, as_array=True, mask=mask, random_state=actual_rng)
        sampled = points.subsample(0.5, as_array=True, mask=mask, random_state=42)

        # Compare exact equality
        np.testing.assert_array_equal(indices[0], expected_indices)
        np.testing.assert_array_equal(sampled, values[expected_seed_indices])
        assert sampled.dtype == values.dtype
        assert actual_rng.integers(100000) == expected_rng.integers(100000)
        assert points.ds.equals(original)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestPointSubsampleChunked:
    """Test module for point subsample() across Dask and multiprocessing inputs.

    We check lazy point values, masks, file outputs, and backend validation.
    """

    @pytest.mark.parametrize("subsample", [1, 5, 0.25, 0.001])
    @pytest.mark.parametrize("return_indices", [False, True])
    @pytest.mark.parametrize("mask_form", ["none", "array", "dask", "vector"])
    def test_subsample__lazy_point_values(self, subsample: int | float, return_indices: bool, mask_form: str) -> None:
        """Checks that lazy point sampling gathers the requested values in the same seeded order as NumPy."""

        # Give several points the same labels so returned indexes must refer to row positions
        import dask_geopandas as dgpd

        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        values = np.arange(40, dtype=float) - 1
        values[::9] = np.nan
        values[2] = np.inf
        dataframe = gpd.GeoDataFrame(
            {"height": values}, geometry=gpd.points_from_xy(np.arange(40), np.zeros(40)), crs=32632
        )
        dataframe.index = np.arange(40) % 3
        _register_dask_pointcloud_accessor()
        lazy = dgpd.from_geopandas(dataframe, npartitions=6, sort=False)
        assert not lazy.pc.is_loaded

        # Restrict the population before applying counts or fractions, without changing the point data
        eligible = np.ones(40, dtype=bool)
        mask: Any = None
        if mask_form == "array":
            eligible = np.arange(40) % 4 != 0
            mask = eligible
        elif mask_form == "dask":
            import dask.array as da

            eligible = np.arange(40) % 4 != 0
            mask = da.from_array(eligible, chunks=7)
        elif mask_form == "vector":
            eligible = (np.arange(40) >= 10) & (np.arange(40) < 30)
            mask = gpd.GeoDataFrame(geometry=[box(9.5, -0.5, 29.5, 0.5)], crs=dataframe.crs)
        expected_values = np.ma.array(values, mask=~eligible)

        # Compare exact random order and generator advancement with the established finite NumPy sampler
        actual_rng = np.random.default_rng(42)
        expected_rng = np.random.default_rng(42)
        expected_indices = _subsample_numpy(expected_values, subsample, return_indices=True, random_state=expected_rng)
        expected = expected_indices if return_indices else values[expected_indices]
        series_type = type(lazy["height"])
        with patch.object(series_type, "compute", autospec=True, side_effect=series_type.compute) as compute:
            result = lazy.pc.subsample(
                subsample,
                return_indices=return_indices,
                as_array=True,
                random_state=actual_rng,
                mask=mask,
            )

        # Only row-count summaries may be collected as Series; the full original data column stays partitioned
        assert all(call.args[0].name != "height" for call in compute.call_args_list)
        np.testing.assert_array_equal(result, expected)
        assert not lazy.pc.is_loaded
        assert actual_rng.integers(100000) == expected_rng.integers(100000)
        if return_indices:
            assert isinstance(result, tuple) and isinstance(result[0], np.ndarray)
        else:
            assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("loaded", [False, True])
    def test_subsample__point_output_uses_geopackage(self, loaded: bool, tmp_path: Path) -> None:
        """Checks that multiprocessing writes complete selected point rows to an unloaded GeoPackage output."""

        # Add an attribute beside the active values so the output must keep every point column
        values = np.arange(40, dtype=np.float64)
        points = gu.PointCloud.from_xyz(values, np.zeros(40), values, crs=32633)
        points.ds["label"] = [f"point-{index}" for index in range(40)]
        source_file = tmp_path / "point-source.gpkg"
        points.to_file(source_file, index=False)
        source = points if loaded else gu.PointCloud(source_file, data_column="z")
        output_file = tmp_path / f"point-sample-{loaded}.gpkg"

        # Select more rows than one seven-row partition and write complete point rows in source order
        expected_indices = points.subsample(
            17,
            return_indices=True,
            as_array=True,
            random_state=42,
            strategy="topk",
        )[0]
        result = source.subsample(
            17,
            random_state=42,
            strategy="topk",
            mp_config=MultiprocConfig(chunks=7, outfile=str(output_file)),
        )

        # Keep the result unopened until requested, then match all source values, attributes, and geometries
        assert isinstance(result, gu.PointCloud)
        assert not result.is_loaded
        assert source.is_loaded == loaded
        expected = points.ds.iloc[np.sort(expected_indices)].reset_index(drop=True)
        gpd.testing.assert_geodataframe_equal(
            result.ds.reset_index(drop=True), expected, check_dtype=False, check_like=True
        )
        assert source.is_loaded == loaded

    def test_subsample__point_accessor_multiprocessing_output(self, tmp_path: Path) -> None:
        """Checks that multiprocessing point subsampling returns a GeoDataFrame for an eager accessor input."""

        # Add a second attribute so the accessor result must keep complete source rows
        values = np.arange(40, dtype=np.float64)
        points = gu.PointCloud.from_xyz(values, np.zeros(40), values, crs=32633)
        points.ds["label"] = [f"point-{index}" for index in range(40)]
        output_file = tmp_path / "accessor-sample.gpkg"

        # Write a sample larger than one seven-row partition through the GeoDataFrame accessor
        result = points.ds.pc.subsample(
            17,
            random_state=42,
            strategy="topk",
            mp_config=MultiprocConfig(chunks=7, outfile=str(output_file)),
        )
        expected_indices = points.subsample(
            17,
            return_indices=True,
            as_array=True,
            random_state=42,
            strategy="topk",
        )[0]

        # Check the GeoDataFrame and all selected columns
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.pc.data_column == "z"
        expected = points.ds.iloc[np.sort(expected_indices)].reset_index(drop=True)
        gpd.testing.assert_geodataframe_equal(
            result.reset_index(drop=True), expected, check_dtype=False, check_like=True
        )

    @pytest.mark.parametrize("suffix", [".las", ".laz"])
    def test_subsample__point_multiproc_las_elevation(self, suffix: str, tmp_path: Path) -> None:
        """Checks that LAS and LAZ point subsampling stores the main data column as elevation."""

        # Build a loaded point cloud with one main value and one numeric auxiliary column
        dataframe = gpd.GeoDataFrame(
            {
                "height": np.arange(20, dtype=np.int16),
                "quality": np.arange(20, dtype=np.uint8) + 100,
            },
            geometry=gpd.points_from_xy(np.arange(20), np.arange(20) + 50),
            crs=32633,
        )
        source = gu.PointCloud(dataframe, data_column="height")
        expected = source.subsample(7, random_state=42).ds
        output_file = tmp_path / f"sampled-points{suffix}"

        # Select the same rows into LAS/LAZ, mapping the active height values to native Z
        result = source.subsample(
            7,
            random_state=42,
            mp_config=MultiprocConfig(chunks=6, outfile=str(output_file)),
        )

        # Keep the result unloaded, then compare its elevations and auxiliary values with the eager sample
        assert output_file.exists() and not result.is_loaded
        assert result.data_column == "Z"
        result.load(columns=["Z", "quality"])
        expected = expected.sort_values("height").reset_index(drop=True)
        np.testing.assert_array_equal(result.geometry.x, expected.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, expected.geometry.y)
        np.testing.assert_array_equal(result.data, expected["height"])
        np.testing.assert_array_equal(result.ds["quality"], expected["quality"])

    def test_subsample__unloaded_point_output_uses_npy(self, tmp_path: Path) -> None:
        """Checks that point values are read and written in partitions without loading their source file."""

        # Save the point source for partitioned reads
        values = np.arange(40, dtype=np.float64)
        points = gu.PointCloud.from_xyz(values, np.zeros(40), values, crs=32633)
        source_file = tmp_path / "point-source.gpkg"
        output_file = tmp_path / "point-values.npy"
        points.to_file(source_file, index=False)
        source = gu.PointCloud(source_file, data_column="z")
        expected = points.subsample(17, as_array=True, random_state=42, strategy="topk")
        assert not source.is_loaded

        # Read seven source rows at a time and write the larger selected output to a NumPy memory map
        result = source.subsample(
            17,
            as_array=True,
            random_state=42,
            strategy="topk",
            mp_config=MultiprocConfig(chunks=7, outfile=str(output_file)),
        )

        # Preserve exact key order and leave the file-backed point cloud unopened
        assert isinstance(result, np.memmap)
        np.testing.assert_array_equal(result, expected)
        assert not source.is_loaded

    @pytest.mark.parametrize("return_indices", [False, True])
    def test_subsample__point_large_output_uses_npy(
        self,
        return_indices: bool,
        tmp_path: Path,
    ) -> None:
        """Checks that large multiprocessing point samples write values or row positions to a NumPy file."""

        # Build finite point values for the multiprocessing path
        values = np.arange(40, dtype=np.float64)
        points = gu.PointCloud.from_xyz(values, np.zeros(40), values, crs=32633)
        output_file = tmp_path / f"point-{return_indices}.npy"
        expected = points.subsample(
            17,
            return_indices=return_indices,
            as_array=True,
            random_state=42,
            strategy="topk",
        )

        # Select more rows than one seven-row partition and write the output without collecting it in memory
        result = points.subsample(
            17,
            return_indices=return_indices,
            as_array=True,
            random_state=42,
            strategy="topk",
            mp_config=MultiprocConfig(chunks=7, outfile=str(output_file)),
        )
        stored = np.load(output_file, mmap_mode="r")

        # Check the stored shape, type, and values
        expected_array = np.asarray(expected)
        expected_shape = (1, 17) if return_indices else (17,)
        assert stored.shape == expected_shape
        assert isinstance(stored, np.memmap)
        np.testing.assert_array_equal(stored, expected_array)
        np.testing.assert_array_equal(np.asarray(result), expected_array)
        if return_indices:
            assert isinstance(result[0], np.memmap)
        else:
            assert isinstance(result, np.memmap)

    def test_subsample__point_topk_cutoff_and_force_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that large lazy point samples use cutoff search unless memory output is forced."""

        import dask_geopandas as dgpd

        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor
        from geoutils.sampling import subsampling as subsampling_module

        # Split finite point values into partitions much smaller than the requested sample
        values = np.arange(40, dtype=np.float64)
        points = gu.PointCloud.from_xyz(values, np.zeros(40), values, crs=32633)
        expected = points.subsample(17, as_array=True, random_state=42, strategy="topk")
        _register_dask_pointcloud_accessor()
        lazy = dgpd.from_geopandas(points.ds, npartitions=6, sort=False)

        # Record the automatic cutoff search while preserving its result
        cutoff = subsampling_module._dask_array_topk_cutoff
        cutoff_calls = []

        def record_cutoff(*args: Any, **kwargs: Any) -> Any:
            """Record one cutoff search for the lazy point values."""

            cutoff_calls.append(1)
            return cutoff(*args, **kwargs)

        monkeypatch.setattr(subsampling_module, "_dask_array_topk_cutoff", record_cutoff)
        result = lazy.pc.subsample(17, as_array=True, random_state=42, strategy="topk")
        forced = lazy.pc.subsample(
            17,
            as_array=True,
            random_state=42,
            strategy="topk",
            force_output_to_memory=True,
        )
        forced_points = lazy.pc.subsample(
            17,
            random_state=42,
            strategy="topk",
            force_output_to_memory=True,
        )

        # Check the automatic and forced memory results
        assert cutoff_calls == [1]
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(forced, expected)
        assert isinstance(forced_points, gpd.GeoDataFrame)
        gpd.testing.assert_geodataframe_equal(
            forced_points,
            points.subsample(17, random_state=42, strategy="topk").ds,
        )
        assert not lazy.pc.is_loaded

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("tiny_fraction", [False, True])
    def test_subsample__point_empty_masks(self, lazy: bool, tiny_fraction: bool) -> None:
        """Checks that empty point populations and fractions rounded to zero return empty values and positions."""

        # Either exclude every point or select one point whose half-sample rounds down to zero
        values = np.arange(6, dtype=np.int16)
        points = gu.PointCloud.from_xyz(values, np.zeros(6), values, crs=32633)
        mask = np.zeros(6, dtype=bool)
        subsample = 1.0
        if tiny_fraction:
            mask[-1] = True
            subsample = 0.5
        expected_sample = points.subsample(subsample, as_array=True, mask=mask, random_state=42)
        expected_indices = points.subsample(
            subsample,
            return_indices=True,
            as_array=True,
            mask=mask,
            random_state=42,
        )
        source = points
        if lazy:
            import dask_geopandas as dgpd

            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor,
            )

            _register_dask_pointcloud_accessor()
            source = dgpd.from_geopandas(points.ds, npartitions=2, sort=False).pc
            assert not source.is_loaded

        # Preserve the ordinary value dtype and the one-dimensional positional-index layout
        sampled = source.subsample(subsample, as_array=True, mask=mask, random_state=42)
        indices = source.subsample(
            subsample,
            return_indices=True,
            as_array=True,
            mask=mask,
            random_state=42,
        )
        assert sampled.size == 0 and sampled.dtype == values.dtype
        assert len(indices) == 1 and indices[0].size == 0
        assert np.issubdtype(indices[0].dtype, np.integer)
        assert np.array_equal(sampled, expected_sample)
        assert np.array_equal(indices, expected_indices)
        if lazy:
            assert not source.is_loaded


class TestRasterSubsample:
    """
    Test module for eager raster subsampling into point clouds and value arrays.

    TestRasterSubsampleChunked further below covers Dask and multiprocessing outputs,
    their loading behavior, and backend validation.
    """

    def test_subsample__band_inputs(self) -> None:
        """Checks that integer, iterable and mapping band inputs select and name the requested values."""

        # Build three bands and pass the selected band numbers as a one-use generator
        values = np.arange(12).reshape((3, 2, 2))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 32633)
        bands = (band for band in (1, 3))
        pointcloud = raster.subsample(1, bands=bands)

        # Preserve both selected values with their default band names
        assert list(pointcloud.ds.columns) == ["b1", "b3", "geometry"]
        np.testing.assert_array_equal(pointcloud["b1"].to_numpy(), values[0].ravel())
        np.testing.assert_array_equal(pointcloud["b3"].to_numpy(), values[2].ravel())

        # Select one band directly, then name and reorder several bands with a mapping
        single_band = raster.subsample(1, bands=2)
        named_bands = raster.subsample(1, bands={"third": 3, "first": 1})
        assert list(single_band.ds.columns) == ["b2", "geometry"]
        assert list(named_bands.ds.columns) == ["third", "first", "geometry"]
        np.testing.assert_array_equal(single_band["b2"].to_numpy(), values[1].ravel())
        np.testing.assert_array_equal(named_bands["third"].to_numpy(), values[2].ravel())
        np.testing.assert_array_equal(named_bands["first"].to_numpy(), values[0].ravel())

    def test_to_pointcloud__same_output_as_subsample(self) -> None:
        """Checks that to_pointcloud(subsample=) returns the exact output of subsample()."""

        # Create three distinct bands so the default comparison covers every raster band
        values = np.arange(36).reshape((3, 3, 4))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 3, 1, 1), 32633)

        # Compare both public entry points row for row with the same deterministic selection
        sampled = raster.subsample(7, random_state=42)
        converted = raster.to_pointcloud(subsample=7, random_state=42)
        assert list(sampled.ds.columns) == ["b1", "b2", "b3", "geometry"]
        assert sampled.data_column == converted.data_column
        assert sampled.ds.equals(converted.ds)

    def test_subsample__accessor_returns_geodataframe(self) -> None:
        """Checks that an eager raster accessor returns a GeoDataFrame by default."""

        # Create an accessor with two bands to check both its output type and default band selection
        values = np.arange(16).reshape((2, 2, 4))
        accessor = gu.RasterAccessor.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 32633)

        # Sample through the accessor and keep every band in the dataframe
        sampled = accessor.rst.subsample(5, random_state=42)
        assert isinstance(sampled, gpd.GeoDataFrame)
        assert list(sampled.columns) == ["b1", "b2", "geometry"]

    @pytest.mark.parametrize(
        "mask_form",
        [
            "array",
            "list",
            "tuple",
            "masked",
            "xarray",
            "vector",
            "vector_accessor",
            "raster",
            "raster_accessor",
        ],
    )
    def test_subsample__raster_masks(self, mask_form: str) -> None:
        """Checks that masks restrict the sampled band and return its exact values and original grid indices."""

        # Use distinct integer bands and a nodata second-band pixel to expose band selection or mask mistakes
        values = np.arange(36, dtype=np.int16).reshape(6, 6)
        data = np.ma.array(np.stack([values + 100, values]), mask=False)
        data.mask[1, 1, 1] = True
        raster = gu.Raster.from_array(data, from_origin(0, 6, 1, 1), crs=32633, nodata=-9999)
        original = raster.data.copy()
        eligible = np.indices(raster.shape)[1] < 3
        mask: Any = eligible.copy()

        # Select the left half of the grid, excluding nodata entries of an explicitly masked boolean array
        if mask_form == "list":
            mask = mask.tolist()
        elif mask_form == "tuple":
            mask = tuple(tuple(row) for row in mask.tolist())
        elif mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[2, 1] = True
            eligible[2, 1] = False
        elif mask_form == "xarray":
            mask = xr.DataArray(mask[None], dims=("band", "row", "column"))
        elif mask_form in {"vector", "vector_accessor"}:
            vector_mask = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 6)], crs=raster.crs)
            mask = gu.Vector(vector_mask) if mask_form == "vector" else vector_mask
        elif mask_form == "raster":
            mask = gu.Raster.from_array(mask, raster.transform, raster.crs)
        elif mask_form == "raster_accessor":
            mask = gu.RasterAccessor.from_array(mask, raster.transform, raster.crs)
        eligible &= ~data.mask[1]

        # Draw the expected top-k samples to check the count, order and generator advancement
        eligible_values = np.ma.array(values, mask=~eligible)
        expected_rng = np.random.default_rng(42)
        expected_indices = _subsample_numpy(eligible_values, 0.5, return_indices=True, random_state=expected_rng)
        expected_seed_indices = _subsample_numpy(eligible_values, 0.5, return_indices=True, random_state=42)
        actual_rng = np.random.default_rng(42)
        indices = _subsample_values(raster, 0.5, band=2, return_indices=True, mask=mask, random_state=actual_rng)
        sampled = _subsample_values(raster, 0.5, band=2, mask=mask, random_state=42)

        # Check that sampled integers match the selected band and the source data and nodata mask are unchanged
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(sampled, values[expected_seed_indices])
        assert sampled.dtype == values.dtype
        assert actual_rng.integers(100000) == expected_rng.integers(100000)
        np.testing.assert_array_equal(raster.data.data, original.data)
        np.testing.assert_array_equal(raster.data.mask, original.mask)

    def test_subsample__keep_nodata(self) -> None:
        """Checks that skip_nodata=False includes nodata cells and a user mask restricts the grid properly."""

        # Create a synthetic raster with two finite cells and keep only the first five columns as a mask
        values = np.ma.array(np.arange(24, dtype=np.int16).reshape(4, 6), mask=True)
        values.mask[0, 0] = False
        values.mask[3, 4] = False
        raster = gu.Raster.from_array(values, from_origin(0, 4, 1, 1), crs=32633, nodata=-9999)
        allowed = np.indices(raster.shape)[1] < 5

        # Subsample with/without nodata skipping
        all_indices = _subsample_values(raster, 1, return_indices=True, skip_nodata=False, mask=allowed)
        finite_indices = _subsample_values(raster, 1, return_indices=True, mask=allowed)

        # Check the flattened output
        expected_flat = np.flatnonzero(allowed)
        np.testing.assert_array_equal(np.ravel_multi_index(all_indices, raster.shape), expected_flat)
        np.testing.assert_array_equal(np.ravel_multi_index(finite_indices, raster.shape), np.array([0, 22]))


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestRasterSubsampleChunked:
    """
    Test module for subsample() of rasters across eager, Dask, and multiprocessing backends.

    Dask outputs must stay lazy until explicitly computed. Multiprocessing inputs and point outputs must stay unloaded,
    and both chunked backends must return the same values as an eager conversion.
    """

    @pytest.mark.parametrize("backend", ["dask", "multiprocessing"])
    def test_subsample__large_sample_uses_cutoff(
        self,
        backend: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Checks that both chunked backends use cutoff search for a sample size larger than an input raster chunk."""

        from geoutils.sampling import subsampling as subsampling_module

        # Keep only high random keys to test the fallback when the initial cutoff estimate misses the sample
        cell_indices = np.arange(64 * 64, dtype=np.int32)
        keys = np.asarray(_splitmix64(np.uint64(42) ^ cell_indices.astype(np.uint64)), dtype=np.uint64)
        valid = keys >> np.uint64(56) >= 128
        values = np.ma.masked_array(cell_indices.reshape((64, 64)), mask=~valid.reshape((64, 64)), fill_value=-9999)
        source_file = tmp_path / "cutoff-source.tif"
        output_file = tmp_path / "multiprocessing-cutoff.gpkg"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 64, 1, 1), 32633, nodata=-9999)
        raster.to_file(source_file)
        expected_values = raster.subsample(65, random_state=42, as_array=True)

        # Record the cutoff search
        cutoff_name = "_dask_array_topk_cutoff" if backend == "dask" else "_multiproc_array_topk_cutoff"
        cutoff = getattr(subsampling_module, cutoff_name)
        cutoff_calls = []

        def record_cutoff(*args: Any, **kwargs: Any) -> Any:
            """Record one backend cutoff search while preserving its result."""

            cutoff_calls.append(1)
            return cutoff(*args, **kwargs)

        monkeypatch.setattr(subsampling_module, cutoff_name, record_cutoff)

        # Subsample with the requested backend
        if backend == "dask":
            source = open_raster(str(source_file), chunks={"x": 8, "y": 8})
            source_data = source.data
            result = source.rst.subsample(65, random_state=42)
            assert is_dask_dataframe(result) and cutoff_calls == [1]
            computed_values = result.compute()["b1"].to_numpy()
            assert source.data is source_data and not source._in_memory
        else:
            source = gu.Raster(source_file)
            result = source.subsample(
                65,
                random_state=42,
                mp_config=MultiprocConfig(chunks=(8, 8), outfile=str(output_file)),
            )
            assert cutoff_calls == [1]
            assert not source.is_loaded and not result.is_loaded
            computed_values = result.ds["b1"].to_numpy()
            assert not source.is_loaded

        # Compare with the eager result
        np.testing.assert_array_equal(np.sort(computed_values), np.sort(expected_values))

    @pytest.mark.parametrize("backend", ["dask", "multiprocessing"])
    def test_subsample__force_output_to_memory(
        self,
        backend: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Checks that forcing memory output bypasses cutoff search and returns eager raster point rows."""

        from geoutils.sampling import subsampling as subsampling_module

        # Write a source whose 17 selected cells exceed every 3 x 4 input chunk
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "forced-memory-source.tif"
        output_file = tmp_path / "unused-forced-output.gpkg"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633)
        raster.to_file(source_file)
        expected = raster.subsample(17, random_state=42).ds

        # Make cutoff search fail if it is used
        cutoff_name = "_dask_array_topk_cutoff" if backend == "dask" else "_multiproc_array_topk_cutoff"

        def reject_cutoff(*args: Any, **kwargs: Any) -> Any:
            """Fail if forced memory output performs cutoff search."""

            raise AssertionError("Forced memory output must bypass cutoff search.")

        monkeypatch.setattr(subsampling_module, cutoff_name, reject_cutoff)

        # Subsample with the requested backend
        if backend == "dask":
            source = open_raster(str(source_file), chunks={"x": 4, "y": 3})
            result = source.rst.subsample(17, random_state=42, force_output_to_memory=True)
            actual = result
            assert isinstance(result, gpd.GeoDataFrame) and not is_dask_dataframe(result)
            assert not source._in_memory
        else:
            source = gu.Raster(source_file)
            result = source.subsample(
                17,
                random_state=42,
                mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
                force_output_to_memory=True,
            )
            actual = result.ds
            assert result.is_loaded and not source.is_loaded
            assert not output_file.exists()

        # Compare with the eager result
        gpd.testing.assert_geodataframe_equal(
            actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
        )

    def test_subsample__multiproc_reads_each_selected_tile_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a large sample is selected and read one raster tile at a time."""

        from geoutils.multiproc import readers
        from geoutils.sampling import subsampling as raster_subsampling

        # Write unique values across six tiles so the selected points can be checked independently of output order
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "tiled-point-source.tif"
        output_file = tmp_path / "tiled-points.gpkg"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633)
        raster.to_file(source_file)
        source = gu.Raster(source_file)
        chunks = (3, 4)
        expected = raster.subsample(17, random_state=42, strategy="topk", as_array=True)

        # Record the cells passed to every multi-band read while preserving the real file operation
        read_groups: list[NDArray[np.int64]] = []
        read_selected = readers._read_selected_raster_bands

        def count_selected_tiles(source_raster: Any, indexes: Any, bands: list[int], tile_size: Any) -> Any:
            """Record one selected cell group before reading its raster values."""

            read_groups.append(np.asarray(indexes, dtype=np.int64))
            return read_selected(source_raster, indexes, bands, tile_size)

        monkeypatch.setattr(readers, "_read_selected_raster_bands", count_selected_tiles)

        # Reject the small-sample path because 17 points exceed the largest 3 x 4 tile
        def reject_indexed_path(*args: Any, **kwargs: Any) -> Any:
            """Fail if point conversion collects and groups the complete sample indexes."""

            raise AssertionError("Large point samples must use the bounded cutoff path.")

        monkeypatch.setattr(raster_subsampling, "_subsample_raster_from_indices", reject_indexed_path)

        # Convert a spatially scattered sample through the synchronous multiprocessing interface
        result = source.subsample(
            subsample=17,
            random_state=42,
            mp_config=MultiprocConfig(chunks=chunks, outfile=str(output_file)),
        )

        # Check that every group belongs to one tile and that no selected tile is read a second time
        tile_columns = (source.width + chunks[1] - 1) // chunks[1]
        grouped_tile_ids = []
        for indexes in read_groups:
            rows, columns = np.unravel_index(indexes, source.shape)
            tile_ids = (rows // chunks[0]) * tile_columns + columns // chunks[1]
            assert len(np.unique(tile_ids)) == 1
            grouped_tile_ids.append(int(tile_ids[0]))
        assert len(grouped_tile_ids) == len(set(grouped_tile_ids))

        # Confirm the returned point cloud stays unopened until its values are requested for comparison
        assert not source.is_loaded and not result.is_loaded
        np.testing.assert_array_equal(np.sort(result.ds["b1"].to_numpy()), np.sort(expected))
        assert not source.is_loaded

    @pytest.mark.parametrize("skip_nodata", [False, True])
    def test_subsample__multiproc_filters_large_sample_before_auxiliary_reads(
        self, skip_nodata: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a large sample reads auxiliary bands only for cells selected by the cutoff."""

        from geoutils.multiproc import readers

        # Write three bands whose 17 selected cells exceed the largest 3 x 4 tile
        main_values = np.arange(80, dtype=np.int16).reshape((8, 10))
        values = np.stack((main_values, main_values + 100, main_values + 200))
        source_file = tmp_path / f"large-multiband-source-{skip_nodata}.tif"
        output_file = tmp_path / f"large-multiband-points-{skip_nodata}.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)
        source = gu.Raster(source_file)

        # Record the indexes and bands passed to each raster read after the cutoff is known
        grouped_reads = []
        read_selected = readers._read_selected_raster_bands

        def record_grouped_read(source_raster: Any, indexes: Any, bands: list[int], chunks: Any) -> Any:
            """Record cells and bands from one raster read."""

            grouped_reads.append((np.asarray(indexes, dtype=np.int64), bands))
            return read_selected(source_raster, indexes, bands, chunks)

        monkeypatch.setattr(readers, "_read_selected_raster_bands", record_grouped_read)

        # Convert a deterministic sample while requesting all three bands
        result = source.subsample(
            subsample=17,
            skip_nodata=skip_nodata,
            random_state=42,
            bands=[1, 2, 3],
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Pass only the 17 selected indexes to the multiband reader when every cell is eligible
        if not skip_nodata:
            assert sum(len(indexes) for indexes, bands in grouped_reads if bands == [1, 2, 3]) == 17

        # Read the main band to establish eligibility, then pass only 17 indexes to the auxiliary reader
        else:
            assert sum(len(indexes) for indexes, bands in grouped_reads if bands == [2, 3]) == 17
        assert result.point_count == 17
        assert not source.is_loaded and not result.is_loaded

    def test_subsample__multiproc_small_sample_uses_indexed_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a sample no larger than one raster chunk uses the shared indexed conversion path."""

        from geoutils.sampling import subsampling as raster_subsampling

        # Write a raster whose 11 selected cells fit within the largest 3 x 4 chunk
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "small-sample-source.tif"
        output_file = tmp_path / "small-sample-points.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)
        source = gu.Raster(source_file)

        # Record the shared indexed conversion while preserving its result
        indexed_conversion = raster_subsampling._subsample_raster_from_indices
        indexed_calls = []

        def record_indexed_conversion(*args: Any, **kwargs: Any) -> Any:
            """Record whether the indexed conversion builds an array or point cloud."""

            indexed_calls.append(kwargs["as_array"])
            return indexed_conversion(*args, **kwargs)

        monkeypatch.setattr(raster_subsampling, "_subsample_raster_from_indices", record_indexed_conversion)

        # Convert the bounded sample and write its materialized point rows once
        result = source.subsample(
            subsample=11,
            random_state=42,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Return the requested file as an unloaded point cloud after using the shared indexed path
        assert indexed_calls == [False]
        assert output_file.exists()
        assert result.name == str(output_file)
        assert not source.is_loaded and not result.is_loaded

    def test_subsample__multiproc_small_sample_groups_multiband_reads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a small sample reads all requested bands once per selected raster tile."""

        from geoutils.multiproc import readers

        # Write three bands whose 11 selected cells fit within one 3 x 4 chunk of output memory
        main_values = np.arange(80, dtype=np.int16).reshape((8, 10))
        values = np.stack((main_values, main_values + 100, main_values + 200))
        source_file = tmp_path / "small-multiband-source.tif"
        output_file = tmp_path / "small-multiband-points.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)
        source = gu.Raster(source_file)

        # Record each grouped multi-band file read while preserving its values
        grouped_reads = []
        read_selected = readers._read_selected_raster_bands

        def record_grouped_read(source_raster: Any, indexes: Any, bands: list[int], chunks: Any) -> Any:
            """Record the selected cells and bands read in one worker call."""

            grouped_reads.append((np.asarray(indexes, dtype=np.int64), bands))
            return read_selected(source_raster, indexes, bands, chunks)

        monkeypatch.setattr(readers, "_read_selected_raster_bands", record_grouped_read)

        # Convert all three bands through the indexed multiprocessing path
        result = source.subsample(
            subsample=11,
            random_state=42,
            bands=[1, 2, 3],
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Read each selected cell once, with all bands together and no tile split across calls
        assert sum(len(indexes) for indexes, _ in grouped_reads) == 11
        assert all(bands == [1, 2, 3] for _, bands in grouped_reads)
        tile_columns = (source.width + 3) // 4
        read_tile_ids = []
        for indexes, _ in grouped_reads:
            rows, columns = np.unravel_index(indexes, source.shape)
            tile_ids = (rows // 3) * tile_columns + columns // 4
            assert len(np.unique(tile_ids)) == 1
            read_tile_ids.append(int(tile_ids[0]))
        assert len(read_tile_ids) == len(set(read_tile_ids))
        assert not result.is_loaded

    def test_subsample__loaded_multiproc_array_uses_npy(self, tmp_path: Path) -> None:
        """Checks that a large array sample from a loaded raster uses multiprocessing NumPy output."""

        # Request more values than one 2 x 3 tile from data already loaded in memory
        values = np.arange(20).reshape((4, 5))
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 4, 1, 1), 32633)
        output_file = tmp_path / "loaded-values.npy"
        result = source.subsample(
            subsample=7,
            random_state=42,
            strategy="topk",
            as_array=True,
            mp_config=MultiprocConfig(chunks=(2, 3), outfile=str(output_file)),
        )

        # Keep the selected values in the configured memory-mapped NumPy file
        assert isinstance(result, np.memmap)
        assert result.shape == (7,)
        assert output_file.exists()

    def test_subsample__multiproc_batches_small_file_writes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that point rows from several small raster tiles are written in one GeoPackage batch."""

        from geoutils.pointcloud import writing

        # Write a source whose scattered sample reaches several 3 x 4 raster tiles
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "batched-point-source.tif"
        output_file = tmp_path / "batched-points.gpkg"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633)
        raster.to_file(source_file)
        source = gu.Raster(source_file)

        # Record the number of rows in each final GeoPackage write
        written_rows = []
        write_dataframe = writing.pyogrio.write_dataframe

        def record_write(dataframe: Any, *args: Any, **kwargs: Any) -> Any:
            """Record one file batch before writing its rows."""

            written_rows.append(len(dataframe))
            return write_dataframe(dataframe, *args, **kwargs)

        monkeypatch.setattr(writing.pyogrio, "write_dataframe", record_write)

        # Convert 17 scattered cells, which is well below the bounded file-write batch size
        result = source.subsample(
            subsample=17,
            random_state=42,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Check that all selected rows were written together and both wrappers remain unloaded
        assert written_rows == [17]
        assert result.point_count == 17
        assert not source.is_loaded and not result.is_loaded

    @pytest.mark.parametrize("suffix", [".las", ".laz"])
    def test_subsample__raster_multiproc_las_elevation(self, suffix: str, tmp_path: Path) -> None:
        """Checks that LAS and LAZ raster subsampling stores the first selected band as elevation."""

        # Write two raster bands whose selected values and projected coordinates are exactly representable in LAS
        height = np.arange(80, dtype=np.int16).reshape((8, 10))
        values = np.stack((height, height + 100))
        source_file = tmp_path / "las-sample-source.tif"
        output_file = tmp_path / f"raster-sample{suffix}"
        gu.Raster.from_array(values, rio.transform.from_origin(500000, 8600000, 20, 20), 32633).to_file(source_file)
        source = gu.Raster(source_file)
        options = {"subsample": 17, "bands": {"height": 1, "quality": 2}, "random_state": 42}
        expected = source.subsample(**options).ds

        # Select the same cells into LAS/LAZ, mapping height to native Z and quality to an extra dimension
        result = source.subsample(
            **options,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Compare tile-ordered file rows with the eager sample after sorting both by X/Y coordinates
        assert output_file.exists() and not source.is_loaded and not result.is_loaded
        assert result.data_column == "Z"
        result.load(columns=["Z", "quality"])
        expected_order = np.lexsort((expected.geometry.x, expected.geometry.y))
        result_order = np.lexsort((result.geometry.x, result.geometry.y))
        np.testing.assert_allclose(result.geometry.x.iloc[result_order], expected.geometry.x.iloc[expected_order])
        np.testing.assert_allclose(result.geometry.y.iloc[result_order], expected.geometry.y.iloc[expected_order])
        np.testing.assert_array_equal(result.data[result_order], expected["height"].iloc[expected_order])
        np.testing.assert_array_equal(result.ds["quality"].iloc[result_order], expected["quality"].iloc[expected_order])

    @pytest.mark.parametrize("backend", ["dask", "multiprocessing"])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_subsample__empty_output(self, backend: str, as_array: bool, tmp_path: Path) -> None:
        """Checks that both chunked backends return empty array or point output without loading the source."""

        # Create an empty input
        source_file = tmp_path / "empty-source.tif"
        output_file = tmp_path / ("empty-values.npy" if as_array else "empty-points.gpkg")
        values = np.ma.masked_all((20, 20), dtype=np.int16)
        values.data.fill(-9999)
        gu.Raster.from_array(values, rio.transform.from_origin(0, 20, 1, 1), 32633, nodata=-9999).to_file(source_file)

        # Subsample with the requested backend
        if backend == "dask":
            source = open_raster(str(source_file), chunks={"x": 5, "y": 5})
            result = source.rst.subsample(1, random_state=42, as_array=as_array)
        else:
            source = gu.Raster(source_file)
            with MpCluster({"nb_workers": 2}) as cluster:
                result = source.subsample(
                    1,
                    random_state=42,
                    as_array=as_array,
                    mp_config=MultiprocConfig(chunks=(5, 5), outfile=str(output_file), cluster=cluster),
                )

        # Check the empty output and loading behavior
        if as_array:
            assert isinstance(result, np.ndarray)
            assert result.shape == (0,)
        elif backend == "dask":
            assert is_dask_dataframe(result)
            assert result.compute().empty
        else:
            assert not result.is_loaded
            assert result.point_count == 0
            assert not result.is_loaded
            assert list(result.ds.columns) == ["b1", "geometry"]
            assert result.is_loaded
        if backend == "dask":
            assert not source._in_memory
        else:
            assert not source.is_loaded


class TestSubsampleErrors:
    """Test module for errors raised by point and raster subsampling."""

    def test_subsample__error_indices_without_array(self) -> None:
        """Checks that point and raster subsampling only return source positions as explicit array outputs."""

        # Build one source of each spatial type supported by subsample()
        values = np.arange(6)
        points = gu.PointCloud.from_xyz(values, np.zeros(6), values, crs=32633)
        raster = gu.Raster.from_array(values.reshape(2, 3), from_origin(0, 2, 1, 1), crs=32633)

        # Require as_array=True because positional indexes cannot be represented by point output
        for source in (points, raster):
            with pytest.raises(ValueError, match="requires ``as_array=True``"):
                source.subsample(1, return_indices=True)

    @pytest.mark.parametrize("wrong_count", [False, True])
    def test_subsample__error_invalid_point_mask(self, wrong_count: bool) -> None:
        """Checks that point subsampling rejects non-boolean masks, or masks with a different shape."""

        # We raise an error for non-boolean masks, or array masks with a wrong shape
        values = np.arange(6)
        points = gu.PointCloud.from_xyz(values, np.zeros(6), values, crs=32633)
        mask = np.ones(5, dtype=bool) if wrong_count else np.ones(6, dtype=int)
        with pytest.raises(
            ValueError, match="Argument ``mask`` must be boolean and contain one value per input location"
        ):
            points.subsample(1, mask=mask)

    @pytest.mark.parametrize("mask_form", ["numeric", "different_shape", "flat", "different_grid"])
    def test_subsample__error_invalid_raster_mask(self, mask_form: str) -> None:
        """Checks that raster masks must contain booleans on the same two-dimensional grid as the source."""

        # Arrays with the same pixel count still need the source shape; spatial masks also need matching coordinates
        raster = gu.Raster.from_array(np.ones((3, 4)), from_origin(0, 3, 1, 1), crs=32633)
        mask: Any = np.ones(raster.shape, dtype=int)
        message = "Argument ``mask`` must be boolean"
        if mask_form == "different_shape":
            mask = np.ones((2, 6), dtype=bool)
            message = "match the support grid"
        elif mask_form == "flat":
            mask = np.ones(12, dtype=bool)
            message = "match the support grid"
        elif mask_form == "different_grid":
            mask = gu.Raster.from_array(np.ones(raster.shape, dtype=bool), from_origin(1, 3, 1, 1), raster.crs)
            message = "does not share the selected support grid"

        # Check that invalid masks are rejected
        with pytest.raises(ValueError, match=message):
            _subsample_values(raster, 1, mask=mask)

    def test_subsample__error_reserved_column_name(self) -> None:
        """Checks that a point data column cannot replace the geometry column."""

        # Create two bands that would otherwise produce a point dataframe with a reserved column name
        values = np.arange(8).reshape((2, 2, 2))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 32633)

        # Reject the reserved name before constructing the dataframe
        with pytest.raises(ValueError, match="must be unique"):
            raster.subsample(1, bands={"geometry": 1})

    @pytest.mark.parametrize("source_kind", ["point", "raster"])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_subsample__error_multiproc_with_dask(self, source_kind: str, as_array: bool, tmp_path: Path) -> None:
        """Checks that point and raster subsampling cannot combine Dask with multiprocessing."""

        # Create a Dask point or raster input
        values = np.arange(40, dtype=np.float64)
        if source_kind == "point":
            dgpd = pytest.importorskip("dask_geopandas")
            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor as register_accessor,
            )

            points = gu.PointCloud.from_xyz(values, np.zeros(40), values, crs=32633)
            register_accessor()
            source = dgpd.from_geopandas(points.ds, npartitions=6, sort=False).pc
            chunks: int | tuple[int, int] = 7
        else:
            da = pytest.importorskip("dask.array")
            lazy_values = da.from_array(values.reshape(5, 8), chunks=(2, 3))
            raster = gu.RasterAccessor.from_array(lazy_values, from_origin(0, 5, 1, 1), 32633)
            source = raster.rst
            chunks = (2, 3)
        suffix = ".npy" if as_array else ".gpkg"
        config = MultiprocConfig(chunks=chunks, outfile=str(tmp_path / f"{source_kind}-sample{suffix}"))

        # Check that multiprocessing is rejected
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            source.subsample(17, as_array=as_array, random_state=42, strategy="topk", mp_config=config)
        assert not source.is_loaded
