"""Test internal raster and point cloud subsampling tools."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils import open_raster
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.raster.array import get_mask_from_array
from geoutils.sampling.subsampling import (
    _sample_valid_indices,
    _subsample_numpy,
)


class TestSubsample:
    """
    Base tests for eager subsampling with one-, two-, and three-dimensional masked arrays.

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

        # Check that 1 returns every available value in the original order
        random_values = _subsample_numpy(array, subsample=1)
        assert np.all(np.sort(random_values) == array[~array.mask])

        random_values_2 = _subsample_numpy(array, subsample=1)
        assert np.array_equal(random_values, random_values_2)

        # Check that a fraction between 0 and 1 returns the right amount of valid values
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


class TestPointSubsample:
    """
    Checks subsample() on point values.

    We only check behaviour with input masks on point data here. Other tests are done directly in TestSubsample, or in
    TestSubsampleChunked.
    """

    @pytest.mark.parametrize("mask_form", ["array", "masked", "xarray", "vector", "raster", "pointcloud"])
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
        if mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[7] = True
            eligible[7] = False
        elif mask_form == "xarray":
            mask = xr.DataArray(mask.reshape(6, 6), dims=("row", "column"))
        elif mask_form == "vector":
            mask = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 2.5, 5.5)], crs=points.crs)
        elif mask_form == "raster":
            mask = gu.Raster.from_array((x < 3)[::-1], from_origin(0, 5, 1, 1), points.crs)
        elif mask_form == "pointcloud":
            mask = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), eligible, crs=points.crs)

        # Draw expected sample
        expected_rng = np.random.default_rng(42)
        expected_indices = expected_rng.choice(np.flatnonzero(eligible), int(eligible.sum() * 0.5), replace=False)
        actual_rng = np.random.default_rng(42)
        indices = points.subsample(0.5, return_indices=True, mask=mask, random_state=actual_rng)
        sampled = points.subsample(0.5, mask=mask, random_state=42)

        # Compare exact equality
        np.testing.assert_array_equal(indices[0], expected_indices)
        np.testing.assert_array_equal(sampled, values[expected_indices])
        assert sampled.dtype == values.dtype
        assert actual_rng.integers(100000) == expected_rng.integers(100000)
        assert points.ds.equals(original)

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


class TestRasterSubsample:
    """Checks subsample() on raster values and masks.

    Selected bands return exact values and original grid indexes. Empty masks return the expected output shape and type.
    Nonempty Dask and Multiproc comparisons are covered in TestSubsampleChunked.
    """

    @pytest.mark.parametrize("mask_form", ["array", "masked", "xarray", "vector", "raster"])
    def test_subsample__raster_masks(self, mask_form: str) -> None:
        """Checks that masks restrict the sampled band and return its exact values and original grid indices."""

        # Use distinct integer bands and a nodata second-band cell to expose band selection or mask mistakes
        values = np.arange(36, dtype=np.int16).reshape(6, 6)
        data = np.ma.array(np.stack([values + 100, values]), mask=False)
        data.mask[1, 1, 1] = True
        raster = gu.Raster.from_array(data, from_origin(0, 6, 1, 1), crs=32633, nodata=-9999)
        original = raster.data.copy()
        eligible = np.indices(raster.shape)[1] < 3
        mask: Any = eligible.copy()

        # Select the left half of the grid, excluding nodata entries of an explicitly masked boolean array
        if mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[2, 1] = True
            eligible[2, 1] = False
        elif mask_form == "xarray":
            mask = xr.DataArray(mask[None], dims=("band", "row", "column"))
        elif mask_form == "vector":
            mask = gpd.GeoDataFrame(geometry=[box(0, 0, 3, 6)], crs=raster.crs)
        elif mask_form == "raster":
            mask = gu.Raster.from_array(mask, raster.transform, raster.crs)
        eligible &= ~data.mask[1]

        # Draw from original flat grid positions to check the count, random order and generator advancement
        expected_rng = np.random.default_rng(42)
        expected_flat = expected_rng.choice(np.flatnonzero(eligible), int(eligible.sum() * 0.5), replace=False)
        expected_indices = np.unravel_index(expected_flat, raster.shape)
        actual_rng = np.random.default_rng(42)
        indices = raster.subsample(0.5, band=2, return_indices=True, mask=mask, random_state=actual_rng)
        sampled = raster.subsample(0.5, band=2, mask=mask, random_state=42)

        # Check that sampled integers match the selected band and the source data and nodata mask are unchanged
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(sampled, values[expected_indices])
        assert sampled.dtype == values.dtype
        assert actual_rng.integers(100000) == expected_rng.integers(100000)
        np.testing.assert_array_equal(raster.data.data, original.data)
        np.testing.assert_array_equal(raster.data.mask, original.mask)

    @pytest.mark.parametrize("mask_form", ["numeric", "different_shape", "flat", "different_grid"])
    def test_subsample__error_invalid_raster_mask(self, mask_form: str) -> None:
        """Checks that raster masks must contain booleans on the same two-dimensional grid as the source."""

        # Arrays with the same cell count still need the source shape; spatial masks also need matching coordinates
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

        # Fail during mask validation instead of sampling a silently coerced or reshaped population
        with pytest.raises(ValueError, match=message):
            raster.subsample(1, mask=mask)


class TestSubsampleChunked:
    """Checks subsample() across eager, Dask and Multiproc inputs.

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

        pytest.importorskip("dask")
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

    @pytest.mark.parametrize("subsample", [1, 5, 0.25, 0.001])
    @pytest.mark.parametrize("return_indices", [False, True])
    @pytest.mark.parametrize("mask_form", ["none", "array", "vector"])
    def test_subsample__lazy_point_values(self, subsample: int | float, return_indices: bool, mask_form: str) -> None:
        """Checks that lazy point sampling gathers the requested values in the same seeded order as NumPy."""

        # Give several points the same labels so returned indexes must refer to row positions
        import_optional("dask_geopandas")
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
            result = lazy.pc.subsample(subsample, return_indices=return_indices, random_state=actual_rng, mask=mask)

        # Only row-count summaries may be collected as Series; the full original data column stays partitioned
        assert all(call.args[0].name != "height" for call in compute.call_args_list)
        np.testing.assert_array_equal(result, expected)
        assert not lazy.pc.is_loaded
        assert actual_rng.integers(100000) == expected_rng.integers(100000)
        if return_indices:
            assert isinstance(result, tuple) and isinstance(result[0], np.ndarray)
        else:
            assert isinstance(result, np.ndarray)

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
        expected_sample = points.subsample(subsample, mask=mask, random_state=42)
        expected_indices = points.subsample(subsample, return_indices=True, mask=mask, random_state=42)
        source = points
        if lazy:
            dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor,
            )

            _register_dask_pointcloud_accessor()
            source = dgpd.from_geopandas(points.ds, npartitions=2, sort=False).pc
            assert not source.is_loaded

        # Preserve the ordinary value dtype and the one-dimensional positional-index layout
        sampled = source.subsample(subsample, mask=mask, random_state=42)
        indices = source.subsample(subsample, return_indices=True, mask=mask, random_state=42)
        assert sampled.size == 0 and sampled.dtype == values.dtype
        assert len(indices) == 1 and indices[0].size == 0
        assert np.issubdtype(indices[0].dtype, np.integer)
        assert np.array_equal(sampled, expected_sample)
        assert np.array_equal(indices, expected_indices)
        if lazy:
            assert not source.is_loaded

    @pytest.mark.parametrize("strategy", ["sequential", "topk"])
    @pytest.mark.parametrize("shape", [(30,), (5, 6)])
    def test_sample_valid_indices__boolean_mask(
        self, strategy: Literal["sequential", "topk"], shape: tuple[int, ...]
    ) -> None:
        """Checks that boolean False locations are excluded and topk preserves positions across array chunks."""

        # Make both accepted and excluded positions occur in every chunk
        import_optional("dask")
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

    @pytest.mark.parametrize("backend", ["numpy", "dask", "multiprocessing"])
    @pytest.mark.parametrize("tiny_fraction", [False, True])
    @pytest.mark.parametrize("strategy", ["sequential", "topk"])
    def test_subsample__raster_empty_masks(
        self, backend: str, tiny_fraction: bool, strategy: Literal["sequential", "topk"], tmp_path: Path
    ) -> None:
        """Checks that empty raster samples have the source value dtype and expected index dimensions."""

        # Select no cells or one cell whose half-sample rounds to zero, crossing several worker tiles
        values = np.arange(24, dtype=np.int16).reshape(4, 6)
        raster = gu.Raster.from_array(values, from_origin(0, 4, 1, 1), crs=32633)
        mask = np.zeros(raster.shape, dtype=bool)
        subsample = 1.0
        if tiny_fraction:
            mask[-1, -1] = True
            subsample = 0.5
        expected_sample = raster.subsample(subsample, mask=mask, random_state=42, strategy=strategy)
        expected_indices = raster.subsample(
            subsample, return_indices=True, mask=mask, random_state=42, strategy=strategy
        )

        # Use an unloaded file for multiprocessing and the usual lazy accessor for Dask
        source: Any = raster
        config = None
        if backend != "numpy":
            path = tmp_path / "empty_sample.tif"
            raster.to_file(path)
            if backend == "dask":
                import_optional("dask")
                source = open_raster(str(path), chunks={"x": 3, "y": 2}).rst
            else:
                source = gu.Raster(path)
                config = MultiprocConfig(chunks=(2, 3))

        # Return empty samples and two empty coordinate arrays without changing the source value dtype
        sampled = source.subsample(subsample, mask=mask, random_state=42, strategy=strategy, mp_config=config)
        indices = source.subsample(
            subsample, return_indices=True, mask=mask, random_state=42, strategy=strategy, mp_config=config
        )
        assert sampled.size == 0 and sampled.dtype == np.dtype(source.dtype)
        assert len(indices) == 2 and all(index.size == 0 for index in indices)
        assert all(np.issubdtype(index.dtype, np.integer) for index in indices)
        assert np.array_equal(sampled, expected_sample)
        assert np.array_equal(indices, expected_indices)
        if backend == "dask":
            da = pytest.importorskip("dask.array")
            assert isinstance(source.data, da.Array) and not source._obj._in_memory
        if backend == "multiprocessing":
            assert not source.is_loaded

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

        # Select all eligible cells so the expected population is independent of random draws and worker tile order
        indices = source.subsample(1, band=selected_band, return_indices=True, mask=mask, mp_config=config)
        sampled = source.subsample(1, band=selected_band, mask=mask, mp_config=config)
        assert len(indices[0]) == eligible.sum()
        assert np.all(eligible[indices])
        np.testing.assert_array_equal(sampled, values[indices])
        assert sampled.dtype == values.dtype
        assert source.bands == original_bands and not source.is_loaded

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

        import_optional("dask")
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

        # Leave the boolean mask file unloaded so workers also read its cells by tile
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
                indices = source.subsample(0.25, return_indices=True, **options)
                sampled = source.subsample(0.25, **options)

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
