"""Tests for sampling raster cell and point row pairs without loading unnecessary data."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu


@pytest.fixture
def raster() -> gu.Raster:
    """Return a finite raster with a small nodata region."""

    array = np.arange(900, dtype=float).reshape(30, 30)
    array[2:5, 4:8] = np.nan
    return gu.Raster.from_array(array, from_origin(0, 30, 2, 3), 32633, nodata=-99999)


class TestRasterPairSampling:
    """Checks pairsample() on raster grids.

    The methods cover sampling strategies, reproducibility, masks, lazy inputs, chunks, and output dtypes.
    """

    @pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
    def test_raster_loglag_strategies_return_pair_dataset(self, raster: gu.Raster, strategy: str) -> None:
        """Checks that every raster strategy returns the requested pairs and labelled values."""

        # 1/ Draw raster pairs with one strategy and fixed distance limits
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

        # 2/ Check the common Xarray layout and that every pair follows the distance and missing data rules
        assert isinstance(pairs, xr.Dataset)
        assert pairs.sizes == {"pair": 200, "endpoint": 2}
        assert set(pairs.data_vars) == {"index", "value", "distance", "row", "column", "x", "y"}
        assert np.all(np.isfinite(pairs.value))
        assert np.all((pairs.distance >= 2) & (pairs.distance <= 40))

    def test_raster_pair_sample_is_reproducible_and_globally_unique(self, raster: gu.Raster) -> None:
        """Checks that a fixed seed returns the same unique raster pairs in the same order."""

        # 1/ Draw the same globally unique sample twice with one seed
        first = raster.pairsample(n_pairs=300, deduplicate="global", random_state=4)
        second = raster.pairsample(n_pairs=300, deduplicate="global", random_state=4)
        indexes = np.sort(first["index"].values, axis=1)

        # 2/ Check exact repeatability and treat reversed endpoint order as the same pair
        assert first.identical(second)
        assert len(np.unique(indexes, axis=0)) == first.sizes["pair"]

    def test_raster_random_xy_and_mask(self, raster: gu.Raster) -> None:
        """Checks that independent raster endpoints stay inside an aligned Boolean mask."""

        # 1/ Allow pairs only in the upper half of the raster
        mask = np.zeros(raster.shape, dtype=bool)
        mask[:15] = True
        pairs = raster.pairsample(n_pairs=100, sampling="random_xy", mask=mask, random_state=8)

        # 2/ Check both endpoints and the sampling method recorded in the result
        assert np.all(pairs["row"] < 15)
        assert pairs.attrs["sampling"] == "random_xy"

    def test_dask_raster_pair_sampling_keeps_source_lazy(self) -> None:
        """Checks that raster pair sampling reads selected Dask cells without loading the source."""

        # 1/ Create a lazy raster with several chunks
        da = pytest.importorskip("dask.array")
        array = np.arange(600, dtype=float).reshape(24, 25)
        raster = gu.RasterAccessor.from_array(
            da.from_array(array, chunks=(6, 5)), from_origin(0, 24, 2, 2), 32633, nodata=None
        )

        # 2/ Draw a fixed number of pairs through the Xarray accessor
        pairs = raster.rst.pairsample(n_pairs=250, random_state=42)

        # 3/ Check the result size and that the complete source is still lazy
        assert pairs.sizes["pair"] == 250
        assert isinstance(raster.data, da.Array)
        assert not raster._in_memory

    def test_raster_pair_sampling_honors_local_chunks_and_dtypes(self) -> None:
        """Checks that nearby Dask pairs stay in one chunk and use the requested number types."""

        # 1/ Create a lazy raster whose row and column chunks have different sizes
        da = pytest.importorskip("dask.array")
        array = np.arange(576, dtype=float).reshape(24, 24)
        raster = gu.RasterAccessor.from_array(
            da.from_array(array, chunks=(6, 8)), from_origin(0, 24, 1, 1), 32633, nodata=None
        )
        # 2/ Request only nearby pairs and smaller output number types
        pairs = raster.rst.pairsample(
            n_pairs=200,
            min_distance=1,
            max_distance=6,
            hybrid_local_fraction=1,
            random_state=7,
            index_dtype=np.int16,
            distance_dtype=np.float32,
        )

        # 3/ Check that both endpoints share a chunk and that output types match the request
        first_chunk = np.column_stack((pairs.row[:, 0] // 6, pairs.column[:, 0] // 8))
        second_chunk = np.column_stack((pairs.row[:, 1] // 6, pairs.column[:, 1] // 8))
        assert np.array_equal(first_chunk, second_chunk)
        assert pairs["index"].dtype == np.int16
        assert pairs["distance"].dtype == np.float32

    def test_raster_pair_sampling_accepts_geodataframe_mask(self, raster: gu.Raster) -> None:
        """Checks that a GeoDataFrame mask keeps both raster endpoints inside its polygon."""

        # 1/ Mask the raster with one polygon that covers only its upper-left area
        mask = gpd.GeoDataFrame(geometry=[box(0, -15, 30, 30)], crs=raster.crs)
        pairs = raster.pairsample(n_pairs=80, mask=mask, max_distance=20, random_state=2)

        # 2/ Check the map coordinates of both endpoints against the polygon bounds
        assert np.all(pairs.x < 30)
        assert np.all(pairs.y > -15)


class TestPointPairSampling:
    """Checks pairsample() on point clouds.

    The methods cover search strategies, masks, source row indexes, output dtypes, and anchor reuse.
    """

    @pytest.mark.parametrize("strategy", ["kdtree", "hashgrid", "nn_logvector"])
    def test_pointcloud_loglag_strategies(self, strategy: str) -> None:
        """Checks that every point strategy returns the requested distances and original rows."""

        # 1/ Create a regular set of points with values that vary in both directions
        y, x = np.mgrid[:20, :20]
        values = np.sin(x.ravel() / 3) + np.cos(y.ravel() / 4)
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        # 2/ Draw pairs with one nearby point search strategy
        pairs = points.pairsample(
            n_pairs=150,
            min_distance=1,
            max_distance=15,
            strategy=strategy,
            anchors_per_round=200,
            nn_tolerance=0.6,
            random_state=3,
        )

        # 3/ Check the pair count, distances, and coordinates from the original rows
        assert pairs.sizes["pair"] == 150
        assert np.all((pairs.distance >= 1) & (pairs.distance <= 15))
        assert np.array_equal(pairs["x"], x.ravel()[pairs["index"]])
        assert np.array_equal(pairs["y"], y.ravel()[pairs["index"]])

    def test_pointcloud_random_pairs_honor_mask_indexes_and_dtypes(self) -> None:
        """Checks that independent point pairs preserve row numbers after missing data and mask filtering."""

        # 1/ Create points with one missing value and allow only the left half
        y, x = np.mgrid[:12, :12]
        values = (x + y).astype(float).ravel()
        values[5] = np.nan
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
        mask = x.ravel() < 6

        # 2/ Draw independent endpoints with smaller output number types
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

        # 3/ Check output types, removal of the missing row, and the mask boundary
        assert pairs["index"].dtype == np.int16
        assert pairs["distance"].dtype == np.float32
        assert not np.any(pairs["index"] == 5)
        assert np.all(pairs.x < 6)

    def test_pointcloud_exact_sampling_reuses_more_anchors_than_points(self) -> None:
        """Checks that exact point searches can reuse first endpoints when one round requests more than exist."""

        # 1/ Create fewer points than the requested number of first endpoints per round
        y, x = np.mgrid[:5, :5]
        points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
        # 2/ Request enough pairs to require reuse during the single allowed round
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

        # 3/ Check that reuse still fills the requested sample
        assert pairs.sizes["pair"] == 50


class TestRasterPairMasking:
    """Checks that raster pair sampling honors every source and user mask.

    The method covers masked integer data and Boolean masks supplied as arrays or rasters.
    """

    @pytest.mark.parametrize("raster_mask", [False, True])
    def test_raster_pairs_exclude_masked_integer_values_and_mask_cells(self, raster_mask: bool) -> None:
        """Checks that masked integer values and masked Boolean cells never enter raster pairs."""

        # 1/ Exclude different cells through the integer data, the mask's own mask, and a false mask value
        data = np.ma.array(np.arange(100, dtype=np.int32).reshape(10, 10), mask=False)
        data.mask[0, 0] = True
        mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
        mask.mask[1, 1] = True
        mask[2, 2] = False
        raster = gu.Raster.from_array(data, from_origin(0, 10, 1, 1), 32633, nodata=-9999)
        selected_mask = raster.from_array(mask, raster.transform, raster.crs) if raster_mask else mask

        # 2/ Draw pairs with either the Boolean array or its Raster form
        pairs = raster.pairsample(n_pairs=200, mask=selected_mask, random_state=3)

        # 3/ Check each endpoint against the combined mask and original integer values
        eligible = ~data.mask & mask.filled(False)
        indexes = pairs["index"].values
        assert pairs.sizes["pair"] == 200
        assert np.all(eligible.ravel()[indexes])
        assert np.array_equal(pairs["value"].values, data.data.ravel()[indexes])
