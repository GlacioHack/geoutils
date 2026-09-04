"""Tests for raster and point cloud pair sampling with bounded memory."""

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


@pytest.mark.parametrize("strategy", ["independent", "anchors", "chunk_anchors", "anchor_batched"])
def test_raster_loglag_strategies_return_pair_dataset(raster: gu.Raster, strategy: str) -> None:
    """Every regular grid strategy should return finite endpoints in the distance range."""

    pairs = raster.sample_pairs(
        n_pairs=200,
        min_distance=2,
        max_distance=40,
        strategy=strategy,
        random_state=42,
        anchors_per_round=100,
        distances_per_anchor=3,
        angles_per_distance=3,
    )

    assert isinstance(pairs, xr.Dataset)
    assert pairs.sizes == {"pair": 200, "endpoint": 2}
    assert set(pairs.data_vars) == {"index", "value", "distance", "row", "column", "x", "y"}
    assert np.all(np.isfinite(pairs.value))
    assert np.all((pairs.distance >= 2) & (pairs.distance <= 40))


def test_raster_pair_sample_is_reproducible_and_globally_unique(raster: gu.Raster) -> None:
    """A fixed seed and global deduplication should provide stable unique undirected pairs."""

    first = raster.sample_pairs(n_pairs=300, deduplicate="global", random_state=4)
    second = raster.sample_pairs(n_pairs=300, deduplicate="global", random_state=4)
    indexes = np.sort(first["index"].values, axis=1)

    assert first.identical(second)
    assert len(np.unique(indexes, axis=0)) == first.sizes["pair"]


def test_raster_random_xy_and_mask(raster: gu.Raster) -> None:
    """Uniform endpoint sampling should honor an aligned eligibility mask."""

    mask = np.zeros(raster.shape, dtype=bool)
    mask[:15] = True
    pairs = raster.sample_pairs(n_pairs=100, sampling="random_xy", mask=mask, random_state=8)

    assert np.all(pairs["row"] < 15)
    assert pairs.attrs["sampling"] == "random_xy"


def test_dask_raster_pair_sampling_keeps_source_lazy() -> None:
    """Only sampled Dask endpoints should be materialized."""

    da = pytest.importorskip("dask.array")
    array = np.arange(600, dtype=float).reshape(24, 25)
    raster = gu.RasterAccessor.from_array(
        da.from_array(array, chunks=(6, 5)), from_origin(0, 24, 2, 2), 32633, nodata=None
    )

    pairs = raster.rst.sample_pairs(n_pairs=250, random_state=42)

    assert pairs.sizes["pair"] == 250
    assert isinstance(raster.data, da.Array)
    assert not raster._in_memory


def test_raster_pair_sampling_honors_local_chunks_and_dtypes() -> None:
    """Sampling within chunks should keep endpoints together and use requested compact dtypes."""

    da = pytest.importorskip("dask.array")
    array = np.arange(576, dtype=float).reshape(24, 24)
    raster = gu.RasterAccessor.from_array(
        da.from_array(array, chunks=(6, 8)), from_origin(0, 24, 1, 1), 32633, nodata=None
    )
    pairs = raster.rst.sample_pairs(
        n_pairs=200,
        min_distance=1,
        max_distance=6,
        hybrid_local_fraction=1,
        random_state=7,
        index_dtype=np.int16,
        distance_dtype=np.float32,
    )

    first_chunk = np.column_stack((pairs.row[:, 0] // 6, pairs.column[:, 0] // 8))
    second_chunk = np.column_stack((pairs.row[:, 1] // 6, pairs.column[:, 1] // 8))
    assert np.array_equal(first_chunk, second_chunk)
    assert pairs["index"].dtype == np.int16
    assert pairs["distance"].dtype == np.float32


def test_raster_pair_sampling_accepts_geodataframe_mask(raster: gu.Raster) -> None:
    """Vector-like masks should restrict both endpoints before pair values are read."""

    mask = gpd.GeoDataFrame(geometry=[box(0, -15, 30, 30)], crs=raster.crs)
    pairs = raster.sample_pairs(n_pairs=80, mask=mask, max_distance=20, random_state=2)

    assert np.all(pairs.x < 30)
    assert np.all(pairs.y > -15)


@pytest.mark.parametrize("strategy", ["kdtree", "hashgrid", "nn_logvector"])
def test_pointcloud_loglag_strategies(strategy: str) -> None:
    """Every irregular point strategy should retain original point indexes and coordinates."""

    y, x = np.mgrid[:20, :20]
    values = np.sin(x.ravel() / 3) + np.cos(y.ravel() / 4)
    points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
    pairs = points.sample_pairs(
        n_pairs=150,
        min_distance=1,
        max_distance=15,
        strategy=strategy,
        anchors_per_round=200,
        nn_tolerance=0.6,
        random_state=3,
    )

    assert pairs.sizes["pair"] == 150
    assert np.all((pairs.distance >= 1) & (pairs.distance <= 15))
    assert np.array_equal(pairs["x"], x.ravel()[pairs["index"]])
    assert np.array_equal(pairs["y"], y.ravel()[pairs["index"]])


def test_pointcloud_random_pairs_honor_mask_indexes_and_dtypes() -> None:
    """Uniform point pairs should preserve original indexes after validity and mask filtering."""

    y, x = np.mgrid[:12, :12]
    values = (x + y).astype(float).ravel()
    values[5] = np.nan
    points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), values, crs=32633)
    mask = x.ravel() < 6

    pairs = points.sample_pairs(
        n_pairs=100,
        sampling="random_xy",
        min_distance=1,
        max_distance=10,
        mask=mask,
        random_state=9,
        index_dtype=np.int16,
        distance_dtype=np.float32,
    )

    assert pairs["index"].dtype == np.int16
    assert pairs["distance"].dtype == np.float32
    assert not np.any(pairs["index"] == 5)
    assert np.all(pairs.x < 6)


def test_pointcloud_exact_sampling_reuses_more_anchors_than_points() -> None:
    """Exact ring sampling should honor a large anchor request per round using replacement."""

    y, x = np.mgrid[:5, :5]
    points = gu.PointCloud.from_xyz(x.ravel(), y.ravel(), (x + y).ravel(), crs=32633)
    pairs = points.sample_pairs(
        n_pairs=50,
        min_distance=1,
        max_distance=5,
        strategy="kdtree",
        anchors_per_round=100,
        attempts_per_anchor=2,
        max_rounds=1,
        random_state=1,
    )

    assert pairs.sizes["pair"] == 50


@pytest.mark.parametrize("raster_mask", [False, True])
def test_raster_pairs_exclude_masked_integer_values_and_mask_cells(raster_mask: bool) -> None:
    """Pair endpoints must respect both data validity and a partially masked Boolean mask."""

    # Keep most cells available while excluding distinct cells through each input
    data = np.ma.array(np.arange(100, dtype=np.int32).reshape(10, 10), mask=False)
    data.mask[0, 0] = True
    mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
    mask.mask[1, 1] = True
    mask[2, 2] = False
    raster = gu.Raster.from_array(data, from_origin(0, 10, 1, 1), 32633, nodata=-9999)
    selected_mask = raster.from_array(mask, raster.transform, raster.crs) if raster_mask else mask

    pairs = raster.sample_pairs(n_pairs=200, mask=selected_mask, random_state=3)

    eligible = ~data.mask & mask.filled(False)
    indexes = pairs["index"].values
    assert pairs.sizes["pair"] == 200
    assert np.all(eligible.ravel()[indexes])
    assert np.array_equal(pairs["value"].values, data.data.ravel()[indexes])
