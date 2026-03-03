"""Test sampling statistical tools."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pytest

import geoutils as gu
from geoutils import open_raster
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.raster.array import get_mask_from_array
from geoutils.stats.sampling import _subsample_numpy


class TestSampling:
    """
    Different examples of 1D to 3D arrays with masked values for testing.
    """

    # Case 1 - 1D array, 1 masked value
    array1D = np.ma.masked_array(np.arange(10), mask=np.zeros(10))
    array1D.mask[3] = True
    assert np.ndim(array1D) == 1
    assert np.count_nonzero(array1D.mask) > 0

    # Case 2 - 2D array, 1 masked value
    array2D = np.ma.masked_array(np.arange(9).reshape((3, 3)), mask=np.zeros((3, 3)))
    array2D.mask[0, 1] = True
    assert np.ndim(array2D) == 2
    assert np.count_nonzero(array2D.mask) > 0

    # Case 3 - 3D array, 1 masked value
    array3D = np.ma.masked_array(np.arange(9).reshape((1, 3, 3)), mask=np.zeros((1, 3, 3)))
    array3D = np.ma.vstack((array3D, array3D + 10))
    array3D.mask[0, 0, 1] = True
    assert np.ndim(array3D) == 3
    assert np.count_nonzero(array3D.mask) > 0

    @pytest.mark.parametrize("array", [array1D, array2D, array3D])
    def test_subsample(self, array: NDArrayNum) -> None:
        """
        Test gu.stats.subsample_array.
        """

        warnings.filterwarnings("ignore", message=".*larger than the number of valid pixels.*", category=UserWarning)

        # Test that subsample > 1 works as expected, i.e. output 1D array, with no masked values, or selected size
        for npts in np.arange(2, np.size(array)):
            random_values = _subsample_numpy(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(random_values.mask) == 0

        # Test if subsample > number of valid values => return all
        random_values = _subsample_numpy(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Test if subsample = 1 => return all valid values
        random_values = _subsample_numpy(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Check that order is preserved for subsample = 1 (no random sampling, simply returns valid mask)
        random_values_2 = _subsample_numpy(array, subsample=1)
        assert np.array_equal(random_values, random_values_2)

        # Test if subsample < 1
        random_values = _subsample_numpy(array, subsample=0.5)
        assert np.size(random_values) == int(np.count_nonzero(~array.mask) * 0.5)

        # Test with optional argument return_indices
        indices = _subsample_numpy(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.count_nonzero(~array.mask) * 0.3)

        # Check that we can pass an integer to fix the random state
        sub42 = _subsample_numpy(array, subsample=10, random_state=42)
        # Check by passing a generator directly
        rng = np.random.default_rng(42)
        sub42_gen = _subsample_numpy(array, subsample=10, random_state=rng)
        # Both should be equal
        assert np.array_equal(sub42, sub42_gen)


class TestSubsampleChunked:

    # Strategies supported by _subsample
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
        """
        Test that subsample behaves consistently across backends:
         - NumPy backend through Raster (in-memory),
         - NumPy backend through Xarray DataArray (in-memory),
         - Dask backend through Xarray accessor (lazy),
         - Multiprocessing backend through Raster (lazy input; eager output by design).

        Notes:
         - "topk" strategy is intended to be chunk-invariant -> outputs should match across backends.
         - "sequential" strategy is chunk/order-dependent -> we do NOT require cross-backend equality,
           but we still validate output properties and determinism for a fixed random_state.
        """

        pytest.importorskip("dask")
        import dask.array as da

        warnings.filterwarnings("ignore", category=UserWarning, message="Subsample value*")

        # Get filepath of on-disk (for laziness) test file
        path_raster = lazy_test_files_tiny[path_index]

        # 1/ Prepare inputs for each backend

        # Base raster input (in-memory -> NumPy backend)
        raster_base = gu.Raster(path_raster)
        raster_base.load()
        assert raster_base.is_loaded

        # Base data array input (in-memory -> NumPy backend through Xarray)
        ds_base = open_raster(path_raster)
        ds_base.load()
        assert ds_base._in_memory

        # Multiprocessing input (keep lazy until mp backend reads)
        raster_mp = gu.Raster(path_raster)
        assert not raster_mp.is_loaded

        # Dask input (lazy chunked xarray)
        ds_dask = open_raster(path_raster, chunks={"x": 10, "y": 10})
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)
        assert ds_dask.data.chunks is not None

        # 2/ Run subsample across backends (fixed seed for determinism)
        seed = 42
        mp_config = MultiprocConfig(chunk_size=10)

        # NumPy backend via Raster
        out_raster = raster_base.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # NumPy backend via Xarray DataArray
        out_xr = ds_base.rst.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # Dask backend via Xarray accessor: should return lazy dask arrays
        out_dask = ds_dask.rst.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # Multiprocessing backend via Raster: output is eager by design, input raster stays lazy
        out_mp = raster_mp.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
            mp_config=mp_config,
        )

        # 3/ Laziness checks

        # Dask input stays unloaded and lazy
        assert not ds_dask._in_memory
        assert isinstance(ds_dask.data, da.Array)

        # Multiprocessing should not load the raster object itself (still points to disk)
        assert not raster_mp.is_loaded

        # Dask output type checks (lazy)
        if return_indices:
            assert isinstance(out_dask, tuple) and len(out_dask) == 2
            assert isinstance(out_dask[0], da.Array)
            assert isinstance(out_dask[1], da.Array)
        else:
            assert isinstance(out_dask, da.Array)

        # 4/ Normalize outputs to comparable NumPy representations

        def _as_numpy(
            out: object,
        ) -> NDArrayNum | tuple[NDArrayNum, NDArrayNum]:
            """Convert backend outputs to NumPy arrays for comparison."""
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

        # 5/ Generic output validity checks (all backends)

        # Mirror _subsample() array selection (band=1 default)
        arr = raster_base.data if raster_base.data.ndim == 2 else raster_base.data[0, :, :]
        assert arr.ndim == 2

        # Mirror _subsample_numpy() validity definition
        mask = get_mask_from_array(arr)  # True where invalid
        n_valid = int(np.count_nonzero(~mask))  # valid pixels

        # Mirror _get_subsample_size_from_user_input()
        if isinstance(subsample, float):
            expected = int(subsample * n_valid)
        else:
            expected = min(int(subsample), n_valid)

        def _check_output(out_np: NDArrayNum | tuple[NDArrayNum, NDArrayNum]) -> None:
            """Validate length, finiteness and index/value consistency."""
            if isinstance(out_np, tuple):
                rr, cc = out_np
                assert rr.shape == cc.shape
                assert rr.ndim == 1 and cc.ndim == 1
                assert len(rr) == expected
                # Indices should be within bounds
                assert np.all((0 <= rr) & (rr < arr.shape[0]))
                assert np.all((0 <= cc) & (cc < arr.shape[1]))
                # Indices must point to finite values
                # Indices must point to valid values per get_mask_from_array
                assert np.all(~mask[rr, cc])
            else:
                assert out_np.ndim == 1
                assert len(out_np) == expected
                assert np.all(np.isfinite(out_np))

        _check_output(out_raster_np)
        _check_output(out_xr_np)
        _check_output(out_dask_np)
        _check_output(out_mp_np)

        # 6/ Backend equivalence logic

        if strategy == "topk":
            # Chunk-invariant strategy: require exact equality across backends.
            assert np.array_equal(out_raster_np, out_xr_np)
            assert np.array_equal(out_raster_np, out_dask_np)
            assert np.array_equal(out_raster_np, out_mp_np)
        else:
            # Sequential: do not require equality across backends (order/chunk dependent),
            # but require determinism per backend for the same seed.
            out_raster_np_2 = _as_numpy(
                raster_base.subsample(
                    subsample=subsample,
                    return_indices=return_indices,
                    random_state=seed,
                    strategy=strategy,
                )
            )
            out_dask_np_2 = _as_numpy(
                ds_dask.rst.subsample(
                    subsample=subsample,
                    return_indices=return_indices,
                    random_state=seed,
                    strategy=strategy,
                )
            )
            out_mp_np_2 = _as_numpy(
                raster_mp.subsample(
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

        # 7/ Indices versus values consistency check for return_indices=True
        if return_indices:
            rr, cc = out_raster_np  # use any backend; for topk they match; for sequential we validate per-backend above
            vals_from_indices = arr[rr, cc]
            # Compare to "values mode" from same backend and same seed/strategy
            vals_raster = raster_base.subsample(
                subsample=subsample,
                return_indices=False,
                random_state=seed,
                strategy=strategy,
            )
            vals_raster_np = _as_numpy(vals_raster)
            assert np.array_equal(np.asarray(vals_from_indices), np.asarray(vals_raster_np))
