"""Tests for sampling raster pixel and point row pairs without loading unnecessary data."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.sampling.pairsampling import _RegularPairSampler


@pytest.fixture
def raster() -> gu.Raster:
    """Return a finite raster with a small nodata region."""

    array = np.arange(900, dtype=float).reshape(30, 30)
    array[2:5, 4:8] = np.nan
    return gu.Raster.from_array(array, from_origin(0, 30, 2, 3), 32633, nodata=-99999)


class TestRasterPairSampling:
    """
    Checks pairsample() on raster grids.

    Dask behavior is covered in TestPairSampleChunked further below.

    This module is checking that:
    - All sampling "strategies" return the correct shape of outputs.
    - User masks, min/max sampling distance and duplicate pairs are all respected.
    """

    @pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
    def test_pairsample__raster_strategies_return_pair_dataset(self, raster: gu.Raster, strategy: str) -> None:
        """Checks that every strategy returns the requested pairs and labelled values."""

        # Draw raster pairs with one strategy and min/max
        pairs = raster.pairsample(
            n_pairs=200,
            min_distance=2,
            max_distance=40,
            strategy=strategy,
            random_state=42,
            anchors_per_round=100,
            distances_per_anchor=3,
            angles_per_distance=3,
        )

        # Check the common Xarray layout and that every pair follows the min/max distance and skips nodata (we
        # must have a finite output)
        assert isinstance(pairs, xr.Dataset)
        assert pairs.sizes == {"pair": 200, "endpoint": 2}
        assert set(pairs.data_vars) == {"index", "value", "distance", "row", "column", "x", "y"}
        assert np.all(np.isfinite(pairs.value))
        assert np.all((pairs.distance >= 2) & (pairs.distance <= 40))
        assert np.array_equal(pairs.value, raster.data.data.ravel()[pairs["index"]])
        expected_distance = np.hypot(np.diff(pairs.x, axis=1), np.diff(pairs.y, axis=1)).ravel()
        np.testing.assert_allclose(pairs.distance, expected_distance)

    def test_pairsample__raster_candidate_batch_limit(self, raster: gu.Raster, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that pairsample() never samples more pairs at once than batch_pairs allows."""

        # Isolate the method that samples pairs before invalid pairs are removed
        candidate_counts = []
        original_candidates = _RegularPairSampler._candidates

        def candidates(sampler: _RegularPairSampler, count: int) -> tuple[NDArrayNum, NDArrayNum]:
            """Record how many pairs were requested, then generate those pairs with the original method."""
            candidate_counts.append(count)
            return original_candidates(sampler, count)

        # During this test, send every call to _candidates() through the recording function above
        monkeypatch.setattr(_RegularPairSampler, "_candidates", candidates)

        # Request 200 pairs while allowing only 37 at once: this forces several calls
        pairs = raster.pairsample(n_pairs=200, batch_pairs=37, strategy="independent", random_state=8)

        # Check that all requested pairs are returned and every candidate batch stays within the limit
        assert pairs.sizes["pair"] == 200
        assert len(candidate_counts) > 1
        assert max(candidate_counts) <= 37

    def test_pairsample__raster_reproducible_and_globally_unique(self, raster: gu.Raster) -> None:
        """Checks that a fixed seed returns the same unique raster pairs in the same order."""

        # Draw the same globally unique sample twice with one seed
        first = raster.pairsample(n_pairs=300, deduplicate="global", random_state=4)
        second = raster.pairsample(n_pairs=300, deduplicate="global", random_state=4)
        indexes = np.sort(first["index"].values, axis=1)

        # Check exact repeatability and treat reversed endpoint order as the same pair
        assert first.identical(second)
        assert len(np.unique(indexes, axis=0)) == first.sizes["pair"]

    def test_pairsample__raster_random_xy_and_mask(self, raster: gu.Raster) -> None:
        """Checks that independent raster endpoints stay inside an aligned boolean mask."""

        # Allow pairs only in the upper half of the raster
        mask = np.zeros(raster.shape, dtype=bool)
        mask[:15] = True
        pairs = raster.pairsample(n_pairs=100, sampling="random_xy", mask=mask, random_state=8)

        # Check both endpoints and the sampling method recorded in the result
        assert np.all(pairs["row"] < 15)
        assert pairs.attrs["sampling"] == "random_xy"

    def test_pairsample__raster_geodataframe_mask(self, raster: gu.Raster) -> None:
        """Checks that a GeoDataFrame mask places both sampled raster endpoints inside its polygon."""

        # Mask the raster with one polygon that covers only its upper-left area
        mask = gpd.GeoDataFrame(geometry=[box(0, -15, 30, 30)], crs=raster.crs)
        pairs = raster.pairsample(n_pairs=80, mask=mask, max_distance=20, random_state=2)

        # Check the map coordinates of both endpoints against the polygon bounds
        assert np.all(pairs.x < 30)
        assert np.all(pairs.y > -15)

    @pytest.mark.parametrize("raster_mask", [False, True])
    def test_pairsample__raster_masked_integers_and_mask_pixels(self, raster_mask: bool) -> None:
        """Checks that masked integer values and masked boolean pixels never enter raster pairs."""

        # Exclude different pixels through the integer data, the mask's own mask, and a false mask value
        data = np.ma.array(np.arange(100, dtype=np.int32).reshape(10, 10), mask=False)
        data.mask[0, 0] = True
        mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
        mask.mask[1, 1] = True
        mask[2, 2] = False
        raster = gu.Raster.from_array(data, from_origin(0, 10, 1, 1), 32633, nodata=-9999)
        selected_mask = raster.from_array(mask, raster.transform, raster.crs) if raster_mask else mask

        # Draw pairs with either the boolean array or its Raster form
        pairs = raster.pairsample(n_pairs=200, mask=selected_mask, random_state=3)

        # Check each endpoint against the combined mask and original integer values
        eligible = ~data.mask & mask.filled(False)
        indexes = pairs["index"].values
        assert pairs.sizes["pair"] == 200
        assert np.all(eligible.ravel()[indexes])
        assert np.array_equal(pairs["value"].values, data.data.ravel()[indexes])


class TestPointPairSampling:
    """
    Checks pairsample() on point clouds.

    - Search strategies return pairs at the requested distances with their original row indexes.
    - Array and spatial masks select the same points.
    - Exact searches cover reused anchors and the upper distance limit.
    """

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid", "nn_logvector"])
    def test_pairsample__point_strategies(self, strategy: str) -> None:
        """Checks that every point strategy returns the requested distances and original rows."""

        # Create a regular set of points with values that vary in both directions
        y, x = np.mgrid[:20, :20]
        values = np.sin(x.ravel() / 3) + np.cos(y.ravel() / 4)
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        # Draw pairs with one nearby point search strategy
        pairs = points.pairsample(
            n_pairs=150,
            min_distance=1,
            max_distance=15,
            strategy=strategy,
            anchors_per_round=200,
            nn_tolerance=0.6,
            random_state=3,
        )

        # Check the pair count, distances, and coordinates from the original rows
        assert pairs.sizes["pair"] == 150
        assert np.all((pairs.distance >= 1) & (pairs.distance <= 15))
        assert np.array_equal(pairs["x"], x.ravel()[pairs["index"]])
        assert np.array_equal(pairs["y"], y.ravel()[pairs["index"]])

    @pytest.mark.parametrize("mask_form", ["array", "masked", "xarray", "dask"])
    def test_pairsample__point_random_pairs_and_mask(self, mask_form: str) -> None:
        """Checks that independent point pairs return original row numbers after nodata and mask filtering."""

        # Create points with one missing value and allow only the left half
        y, x = np.mgrid[:12, :12]
        values = (x + y).astype(float).ravel()
        values[5] = np.nan
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        eligible = x.ravel() < 6
        mask: Any = eligible.copy()

        # Exclude masked entries and accept other array layouts with one boolean value per point
        if mask_form == "masked":
            mask = np.ma.array(mask, mask=False)
            mask.mask[13] = True
            eligible[13] = False
        elif mask_form == "xarray":
            mask = xr.DataArray(mask.reshape(12, 12), dims=("row", "column"))
        elif mask_form == "dask":
            mask = pytest.importorskip("dask.array").from_array(mask, chunks=17)

        # Draw independent endpoints with smaller output number types
        pairs = points.pairsample(
            n_pairs=100,
            sampling="random_xy",
            min_distance=1,
            max_distance=10,
            mask=mask,
            random_state=9,
            index_dtype=np.int16,
            distance_dtype=np.float32,
        )

        # Check output types, removal of the missing row, and the mask boundary
        assert pairs["index"].dtype == np.int16
        assert pairs["distance"].dtype == np.float32
        assert not np.any(pairs["index"] == 5)
        assert np.all(pairs.x < 6)
        assert np.all(eligible[pairs["index"]])

    @pytest.mark.parametrize("mask_form", ["raster", "pointcloud", "vector", "geodataframe"])
    def test_pairsample__point_spatial_masks(self, mask_form: str) -> None:
        """Checks that spatial masks select the same point pairs as their equivalent boolean array."""

        # Select the left half of a point grid, with raster samples at the same integer coordinates
        y, x = np.mgrid[:12, :12]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
        eligible = x < 6
        mask: Any
        if mask_form == "raster":
            mask = gu.Raster.from_array(eligible[::-1], from_origin(0, 11, 1, 1), points.crs)
        elif mask_form == "pointcloud":
            mask = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), eligible.ravel(), crs=points.crs)
        else:
            # Place the first excluded column on the polygon boundary to check that it stays ineligible
            mask = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 6, 11.5)], crs=points.crs)
            if mask_form == "vector":
                mask = gu.Vector(mask)

        # Use identical random draws so both masks must return the same rows, values and distances
        options: dict[str, Any] = {"n_pairs": 100, "sampling": "random_xy", "random_state": 9}
        expected = points.pairsample(mask=eligible.ravel(), **options)
        result = points.pairsample(mask=mask, **options)
        assert result.identical(expected)

    def test_pairsample__point_exact_sampling_reuses_anchors(self) -> None:
        """Checks that exact point searches can reuse first endpoints when one round requests more than exist."""

        # Create fewer points than the requested number of first endpoints per round
        y, x = np.mgrid[:5, :5]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
        # Request enough pairs to require reuse during the single allowed round
        pairs = points.pairsample(
            n_pairs=50,
            min_distance=1,
            max_distance=5,
            strategy="kdtree",
            anchors_per_round=100,
            attempts_per_anchor=2,
            max_rounds=1,
            random_state=1,
        )

        # Check that reuse still fills the requested sample
        assert pairs.sizes["pair"] == 50

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid"])
    def test_pairsample__point_exact_maximum_distance(self, strategy: str) -> None:
        """Checks that exact point searches include pairs at the requested maximum distance."""

        # Use two points whose only nonzero separation is exactly the upper boundary
        points = gu.PointCloud.from_xyz(np.array([0, 1]), np.array([0, 0]), np.array([3, 5]), crs=32633)
        pairs = points.pairsample(
            n_pairs=10, strategy=strategy, n_bins=1, min_distance=0.5, max_distance=1, random_state=2
        )

        # Both exact search methods must find the available pair, with either endpoint order
        assert np.array_equal(pairs.distance, np.ones(10))
        assert np.array_equal(np.sort(pairs["index"], axis=1), np.tile([0, 1], (10, 1)))


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestPairSampleChunked:
    """Checks pairsample() loading behavior and exact results with chunked inputs."""

    @pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
    def test_pairsample__raster_uneven_local_chunks(self, strategy: str) -> None:
        """Checks that local pairs stay in the same actual chunk when interior chunk sizes vary."""

        # Use irregular row and column boundaries that differ from a repeated first-chunk grid
        import dask.array as da

        chunks = ((4, 9, 7), (5, 3, 12))
        eager = np.arange(400, dtype=float).reshape(20, 20)
        array = da.from_array(eager, chunks=chunks)
        raster = gu.RasterAccessor.from_array(array, from_origin(0, 20, 1, 1), 32633)

        # Draw only local pairs so both endpoints must belong to one original Dask chunk
        pairs = raster.rst.pairsample(
            n_pairs=100, strategy=strategy, hybrid_local_fraction=1, min_distance=1, max_distance=6, random_state=4
        )

        # The Dask input stays lazy and pairs are returned as an eager dataset
        assert isinstance(raster.data, da.Array)
        assert not raster._in_memory
        assert not pairs.chunks

        # Find chunks independently from their cumulative boundaries and check each endpoint pair exactly
        rows = np.searchsorted(np.cumsum(chunks[0]), pairs.row, side="right")
        columns = np.searchsorted(np.cumsum(chunks[1]), pairs.column, side="right")
        assert np.array_equal(rows[:, 0], rows[:, 1])
        assert np.array_equal(columns[:, 0], columns[:, 1])
        assert np.array_equal(pairs.value, eager[pairs.row, pairs.column])
        assert pairs.sizes["pair"] == 100

    def test_pairsample__raster_dask_endpoint_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that a Dask chunk is read once for both endpoints of a pair."""

        # Count reads of one source chunk so separate endpoint computations would be visible
        import dask
        import dask.array as da

        reads, candidate_counts = [], []

        @dask.delayed
        def load_values() -> NDArrayNum:
            """Record each source read and return a small raster with finite values."""
            reads.append(1)
            return np.arange(400, dtype=float).reshape(20, 20)

        # Count sampling rounds separately from the initial finite count and final output read
        original_candidates = _RegularPairSampler._candidates

        def candidates(sampler: _RegularPairSampler, count: int) -> tuple[NDArrayNum, NDArrayNum]:
            """Record candidate generation before checking endpoint values."""
            candidate_counts.append(count)
            return original_candidates(sampler, count)

        monkeypatch.setattr(_RegularPairSampler, "_candidates", candidates)
        array = da.from_delayed(load_values(), shape=(20, 20), dtype=float)
        raster = gu.RasterAccessor.from_array(array, from_origin(0, 20, 1, 1), 32633)
        with dask.config.set(scheduler="synchronous"):
            pairs = raster.rst.pairsample(n_pairs=80, strategy="independent", random_state=2)
        dask_candidate_count = len(candidate_counts)
        expected = gu.Raster.from_array(
            np.arange(400, dtype=float).reshape(20, 20), from_origin(0, 20, 1, 1), 32633
        ).pairsample(n_pairs=80, strategy="independent", random_state=2)

        # Expect one count read, one read per round, and one output read regardless of two endpoints
        assert isinstance(raster.data, da.Array) and not raster._in_memory
        assert not pairs.chunks
        xr.testing.assert_equal(pairs, expected)
        assert pairs.sizes["pair"] == 80
        assert len(reads) == dask_candidate_count + 2
        assert np.array_equal(pairs.value, np.asarray(pairs["index"], dtype=float))

    def test_pairsample__raster_dask_source_is_lazy(self) -> None:
        """Checks that raster pair sampling reads selected Dask pixels without loading the source."""

        # Create one lazy raster chunk so sampling order also has an exact eager reference
        import dask.array as da

        array = np.arange(600, dtype=float).reshape(24, 25)
        raster = gu.RasterAccessor.from_array(
            da.from_array(array, chunks=array.shape), from_origin(0, 24, 2, 2), 32633, nodata=None
        )

        # Draw pairs through the Xarray accessor without loading the complete array
        pairs = raster.rst.pairsample(n_pairs=250, hybrid_local_fraction=0, random_state=42)
        expected = gu.Raster.from_array(array, from_origin(0, 24, 2, 2), 32633).pairsample(
            n_pairs=250, hybrid_local_fraction=0, random_state=42
        )
        assert pairs.sizes["pair"] == 250
        assert isinstance(raster.data, da.Array)
        assert not raster._in_memory
        assert not pairs.chunks
        xr.testing.assert_equal(pairs, expected)

    def test_pairsample__raster_dask_local_chunks_and_dtypes(self) -> None:
        """Checks that nearby Dask pairs stay in one chunk and use the requested number types."""

        # Create a lazy raster whose row and column chunks have different sizes
        import dask.array as da

        array = np.arange(576, dtype=float).reshape(24, 24)
        raster = gu.RasterAccessor.from_array(
            da.from_array(array, chunks=(6, 8)), from_origin(0, 24, 1, 1), 32633, nodata=None
        )
        # Request only nearby pairs and smaller output number types
        pairs = raster.rst.pairsample(
            n_pairs=200,
            min_distance=1,
            max_distance=6,
            hybrid_local_fraction=1,
            random_state=7,
            index_dtype=np.int16,
            distance_dtype=np.float32,
        )

        # Check that both endpoints share a chunk and that output types match the request
        assert isinstance(raster.data, da.Array) and not raster._in_memory
        assert not pairs.chunks
        first_chunk = np.column_stack((pairs.row[:, 0] // 6, pairs.column[:, 0] // 8))
        second_chunk = np.column_stack((pairs.row[:, 1] // 6, pairs.column[:, 1] // 8))
        assert np.array_equal(first_chunk, second_chunk)
        assert np.array_equal(pairs.value, array[pairs.row, pairs.column])
        assert pairs["index"].dtype == np.int16
        assert pairs["distance"].dtype == np.float32

    @pytest.mark.parametrize("mask_form", ["vector", "raster", "pointcloud"])
    def test_pairsample__point_masks_reuse_loaded_coordinates(self, mask_form: str) -> None:
        """Checks that spatial masking reads each Dask source partition once during eager pair sampling."""

        # Count reads from three source partitions so a second coordinate load would be visible
        import dask
        import dask_geopandas as dgpd

        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        _register_dask_pointcloud_accessor()
        y, x = np.mgrid[:12, :12]
        dataframe = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633).ds
        reads = []

        def read_partition(partition: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            """Record a source read before returning the point partition."""
            reads.append(1)
            return partition

        lazy_points = dgpd.from_geopandas(dataframe, npartitions=3, sort=False).map_partitions(
            read_partition, meta=dataframe.iloc[:0]
        )

        # Define the same left-half mask, matching integer raster coordinates to the point grid
        mask: Any
        if mask_form == "raster":
            mask = gu.Raster.from_array((x < 6)[::-1], from_origin(0, 11, 1, 1), dataframe.crs)
        elif mask_form == "pointcloud":
            mask = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x < 6).ravel(), crs=dataframe.crs)
        else:
            mask = gpd.GeoDataFrame(geometry=[box(-0.5, -0.5, 5.5, 11.5)], crs=dataframe.crs)

        # Evaluate pairs immediately, reusing the point table that was loaded for the pair search
        with dask.config.set(scheduler="synchronous"):
            pairs = lazy_points.pc.pairsample(n_pairs=100, sampling="random_xy", mask=mask, random_state=9)
        expected = dataframe.pc.pairsample(n_pairs=100, sampling="random_xy", mask=mask, random_state=9)

        assert len(reads) == 3
        assert not lazy_points.pc.is_loaded and not pairs.chunks
        xr.testing.assert_equal(pairs, expected)
        assert pairs.sizes["pair"] == 100
        assert np.all(pairs.x < 6)


class TestPairSampleErrors:
    """Test module for validation errors raised by raster and point pair sampling."""

    @pytest.mark.parametrize("option", ["batch_pairs", "max_rounds", "chunks_per_round", "angles_per_distance"])
    def test_pairsample__error_raster_invalid_batch_controls(self, raster: gu.Raster, option: str) -> None:
        """Checks that zero sampling controls fail before creating batches or indexing empty anchors."""

        with pytest.raises(ValueError, match="controls must be"):
            raster.pairsample(n_pairs=20, **{option: 0})

    @pytest.mark.parametrize("mask_form", ["wrong_count", "numeric", "numeric_pointcloud", "different_coordinates"])
    def test_pairsample__error_point_invalid_masks(self, mask_form: str) -> None:
        """Checks that point masks require boolean values at the same number of ordered source locations."""

        # Use only finite source values so the invalid mask is the sole reason pair sampling fails
        y, x = np.mgrid[:5, :5]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
        mask: Any = np.ones(25, dtype=bool)
        error = "Argument ``mask`` must be boolean and contain one value per input location"
        if mask_form == "wrong_count":
            mask = mask[:-1]
        elif mask_form == "numeric":
            mask = mask.astype(float)
        elif mask_form == "numeric_pointcloud":
            mask = points
            error = "point support mask must contain boolean values"
        else:
            mask = gu.PointCloud.from_xyz(x.ravel() + 1, y.ravel(), mask, crs=points.crs)
            error = "does not share the ordered support coordinates"

        # Report the mask problem before choosing any pair endpoints
        with pytest.raises(ValueError, match=error):
            points.pairsample(n_pairs=10, mask=mask, random_state=9)
