"""Test internal raster and point cloud subsampling tools."""

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
from geoutils.sampling.subsampling import _subsample_numpy


class TestSampling:
    """Tests for _subsample_numpy() with one-, two-, and three-dimensional masked arrays."""

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

    @pytest.mark.parametrize("array", [array1D, array2D, array3D])
    def test_subsample(self, array: NDArrayNum) -> None:
        """Checks that counts, fractions, returned indexes, and random seeds follow the public sampling rules."""

        warnings.filterwarnings("ignore", message=".*larger than the number of valid pixels.*", category=UserWarning)

        # Check every requested count below the input size
        for npts in np.arange(2, np.size(array)):
            random_values = _subsample_numpy(array, subsample=npts)
            assert np.ndim(random_values) == 1
            assert np.size(random_values) == npts
            assert np.count_nonzero(random_values.mask) == 0

        # Check that a count above the available values returns every available value
        random_values = _subsample_numpy(array, subsample=np.size(array) + 3)
        assert np.all(np.sort(random_values) == array[~array.mask])

        # Check that one returns every available value in the original order
        random_values = _subsample_numpy(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        random_values_2 = _subsample_numpy(array, subsample=1)
        assert np.array_equal(random_values, random_values_2)

        # Check that a fraction returns that share of the available values
        random_values = _subsample_numpy(array, subsample=0.5)
        assert np.size(random_values) == int(np.count_nonzero(~array.mask) * 0.5)

        # Check returned indexes against the input dimensions and requested fraction
        indices = _subsample_numpy(array, subsample=0.3, return_indices=True)
        assert np.ndim(indices) == 2
        assert len(indices) == np.ndim(array)
        assert np.ndim(array[indices]) == 1
        assert np.size(array[indices]) == int(np.count_nonzero(~array.mask) * 0.3)

        # Check that an integer seed and the matching NumPy generator select the same values
        sub42 = _subsample_numpy(array, subsample=10, random_state=42)
        rng = np.random.default_rng(42)
        sub42_gen = _subsample_numpy(array, subsample=10, random_state=rng)
        assert np.array_equal(sub42, sub42_gen)


class TestSubsampleChunked:
    """Tests for _subsample() across in-memory, Dask, and multiprocessing paths.

    - test_subsample__backends() checks result size, available values, loading, and repeatability.
    - It also checks exact agreement for `topk`, whose sample must not depend on chunk or tile layout.
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

        pytest.importorskip("dask")
        import dask.array as da

        warnings.filterwarnings("ignore", category=UserWarning, message="Subsample value*")

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
        out_raster = raster_base.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # NumPy through an Xarray accessor
        out_xr = ds_base.rst.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # Dask through an Xarray accessor
        out_dask = ds_dask.rst.subsample(
            subsample=subsample,
            return_indices=return_indices,
            random_state=seed,
            strategy=strategy,
        )

        # Worker processes through Raster
        out_mp = raster_mp.subsample(
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
                # Every returned cell must be available according to the shared raster mask
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

        # 7/ Check that returned indexes select the same values as value mode
        if return_indices:
            rr, cc = out_raster_np
            vals_from_indices = arr[rr, cc]
            vals_raster = raster_base.subsample(
                subsample=subsample,
                return_indices=False,
                random_state=seed,
                strategy=strategy,
            )
            vals_raster_np = _as_numpy(vals_raster)
            assert np.array_equal(np.asarray(vals_from_indices), np.asarray(vals_raster_np))
