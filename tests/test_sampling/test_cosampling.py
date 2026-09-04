"""Tests for sampling two geospatial datasets at common locations."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._typing import NDArrayNum


def _raster(data: NDArrayNum, *, x_origin: float = 0) -> gu.Raster:
    """Create a small raster for cosampling tests."""

    return gu.Raster.from_array(
        data,
        transform=from_origin(x_origin, data.shape[-2], 1, 1),
        crs=32633,
        nodata=-99999,
    )


def test_raster_cosample_uses_joint_validity_mask_and_auxiliary() -> None:
    """Every selected raster cell should be finite in both primaries and the auxiliary."""

    first = np.arange(20, dtype=float).reshape(4, 5)
    second = 10 * first
    auxiliary = 100 * first
    first[0, 0] = np.nan
    second[1, 1] = np.nan
    auxiliary[2, 2] = np.nan
    mask = np.ones(first.shape, dtype=bool)
    mask[3, 3] = False

    result = _raster(first).cosample(
        _raster(second),
        auxiliary={"aux": auxiliary},
        auxiliary_at="self",
        mask=mask,
    )

    assert isinstance(result, gu.CoSampleResult)
    assert len(result) == first.size - 4
    assert np.array_equal(result.other_values, 10 * result.self_values)
    assert np.array_equal(result.auxiliary["aux"], 100 * result.self_values)
    expanded = result.to_support()
    assert expanded["self"].shape == first.shape
    assert np.count_nonzero(np.isfinite(expanded["self"])) == len(result)
    assert result.to_pointcloud().data_column == "self"


def test_cosample_rejects_or_reprojects_an_unaligned_raster() -> None:
    """Grid alignment should be explicit and reproducible."""

    first = _raster(np.arange(12, dtype=float).reshape(3, 4))
    shifted = _raster(np.arange(12, dtype=float).reshape(3, 4), x_origin=1)
    with pytest.raises(ValueError, match="does not share"):
        first.cosample(shifted)

    result = first.cosample(shifted, align="reproject", interpolation="nearest")
    assert len(result) > 0
    assert result.support_kind == "raster"


def test_raster_and_point_cosample_uses_point_support() -> None:
    """A point primary should define support and retain its original indexes."""

    values = np.arange(30, dtype=float).reshape(5, 6)
    values[1, 2] = np.nan
    raster = _raster(values)
    rows = np.array([0, 1, 3, 4])
    columns = np.array([1, 2, 4, 5])
    x, y = raster.ij2xy(rows, columns)
    points = gu.PointCloud.from_xyz(x, y, np.array([4.0, 5.0, 6.0, np.nan]), crs=raster.crs)

    result = raster.cosample(points, interpolation="nearest")

    assert result.support_kind == "pointcloud"
    assert np.array_equal(result.indices[0], [0, 2])
    assert np.array_equal(result.self_values, values[rows[[0, 2]], columns[[0, 2]]])
    assert np.array_equal(result.other_values, [4.0, 6.0])
    assert result.to_arrays(preserve_shape=True)[0].shape == (4,)


def test_pointcloud_cosample_supports_two_primaries_and_point_auxiliary() -> None:
    """Two point clouds and an aligned array should share one ordered point support."""

    x = np.arange(8, dtype=float)
    y = x**2
    first = gu.PointCloud.from_xyz(x, y, x, crs=32633)
    second = gu.PointCloud.from_xyz(x, y, 2 * x, crs=32633)
    auxiliary = 3 * x
    auxiliary[3] = np.nan

    result = first.cosample(second, auxiliary={"weight": auxiliary}, auxiliary_at="self")

    assert len(result) == 7
    assert np.array_equal(result.other_values, 2 * result.self_values)
    assert np.array_equal(result.auxiliary["weight"], 3 * result.self_values)


def test_cosample_selects_raster_bands_and_auxiliary_supports() -> None:
    """Band selection and raw auxiliaries should remain aligned with their named native support."""

    base = np.arange(20, dtype=float).reshape(4, 5)
    first = _raster(np.stack((base, base + 100)))
    second = _raster(np.stack((2 * base, 2 * base + 200)))

    result = first.cosample(
        second,
        band=2,
        other_band=2,
        auxiliary={"first_aux": base + 1, "second_aux": 2 * base + 2},
        auxiliary_at={"first_aux": "self", "second_aux": "other"},
    )

    assert np.array_equal(result.other_values, 2 * result.self_values)
    assert np.array_equal(result.auxiliary["first_aux"], result.self_values - 99)
    assert np.array_equal(result.auxiliary["second_aux"], result.other_values - 198)


def test_pointcloud_cosample_raster_and_vector_mask() -> None:
    """Point object dispatch should interpolate a raster and evaluate a vector directly on points."""

    values = np.arange(30, dtype=float).reshape(5, 6)
    raster = _raster(values)
    rows = np.array([0, 1, 3, 4])
    columns = np.array([1, 2, 4, 5])
    x, y = raster.ij2xy(rows, columns)
    points = gu.PointCloud.from_xyz(x, y, np.arange(4, dtype=float), crs=raster.crs)
    mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 3, 3, 6)], crs=raster.crs))

    result = points.cosample(raster, mask=mask, interpolation="nearest")

    assert np.array_equal(result.indices[0], [0, 1])
    assert np.array_equal(result.self_values, [0, 1])
    assert np.array_equal(result.other_values, values[rows[:2], columns[:2]])


def test_raster_cosample_accepts_geodataframe_mask() -> None:
    """A GeoDataFrame mask should be accepted without requiring a Vector wrapper."""

    first = _raster(np.arange(20, dtype=float).reshape(4, 5))
    geometry = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 4)], crs=first.crs)

    result = first.cosample(first + 1, mask=geometry)

    assert len(result) == 8
    assert np.all(result.coordinates[0] < 2)


def test_cosample_rejects_reserved_auxiliary_names() -> None:
    """Auxiliary names should not be able to replace either primary result."""

    raster = _raster(np.arange(12, dtype=float).reshape(3, 4))

    with pytest.raises(ValueError, match="cannot be"):
        raster.cosample(raster, auxiliary={"self": raster})


def test_cosample_topk_is_independent_of_dask_chunks() -> None:
    """The topk strategy should choose identical cells for different Dask chunk layouts."""

    da = pytest.importorskip("dask.array")
    array = np.arange(63, dtype=float).reshape(7, 9)
    transform = from_origin(0, 7, 1, 1)
    first = gu.RasterAccessor.from_array(
        da.from_array(array, chunks=(2, 4)), transform=transform, crs=32633, nodata=None
    )
    second = gu.RasterAccessor.from_array(
        da.from_array(2 * array, chunks=(4, 3)), transform=transform, crs=32633, nodata=None
    )

    result = first.rst.cosample(second, subsample=12, random_state=42, strategy="topk")
    changed_chunks = gu.RasterAccessor.from_array(
        da.from_array(array, chunks=(4, 3)), transform=transform, crs=32633, nodata=None
    )
    result_changed = changed_chunks.rst.cosample(second, subsample=12, random_state=42, strategy="topk")

    assert np.array_equal(result.indices, result_changed.indices)
    assert np.array_equal(result.self_values, result_changed.self_values)
    assert isinstance(first.data, da.Array)


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("raster_mask", [False, True])
def test_cosample_preserves_masked_integers_and_boolean_masks(lazy: bool, raster_mask: bool) -> None:
    """Masked primaries, auxiliaries and masks must exclude the same support cells."""

    # Give each input a different missing cell so no mask can be silently discarded
    data = np.ma.array(np.arange(30, dtype=np.int32).reshape(5, 6), mask=False)
    data.mask[0, 0] = True
    auxiliary = np.ma.array(2 * data.data, mask=False)
    auxiliary.mask[1, 1] = True
    mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
    mask.mask[2, 2] = True
    mask[3, 3] = False
    first: gu.Raster | gu.RasterAccessor = _raster(data)
    selected_mask = first.from_array(mask, first.transform, first.crs) if raster_mask else mask

    # Exercise lazy grid alignment with the same eager masked auxiliary and mask
    if lazy:
        da = pytest.importorskip("dask.array")
        first = gu.RasterAccessor.from_array(
            da.from_array(data.astype(float).filled(np.nan), chunks=(2, 3)),
            first.transform,
            first.crs,
        ).rst
    result = first.cosample(first, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=selected_mask)

    # Check selected positions as well as values to catch accidentally included masked cells
    expected = ~data.mask & ~auxiliary.mask & mask.filled(False)
    assert len(result) == int(np.count_nonzero(expected))
    assert np.array_equal(np.sort(np.ravel_multi_index(result.indices, data.shape)), np.flatnonzero(expected))
    assert np.array_equal(result.self_values, data.data[result.indices])
    assert np.array_equal(result.auxiliary["aux"], auxiliary.data[result.indices])


def test_point_cosample_preserves_masked_auxiliary_and_mask() -> None:
    """Raw point arrays must retain missing values before common support is sampled."""

    positions = np.arange(5, dtype=float)
    points = gu.PointCloud.from_xyz(positions, positions, positions, crs=32633)
    auxiliary = np.ma.array(np.arange(5), mask=[False, True, False, False, False])
    mask = np.ma.array([True, True, True, False, True], mask=[False, False, True, False, False])

    result = points.cosample(points, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=mask)

    assert np.array_equal(result.indices[0], [0, 4])
    assert np.array_equal(result.auxiliary["aux"], [0, 4])


def test_cosample_mask_keeps_single_row_support() -> None:
    """A one row raster mask must retain both spatial dimensions."""

    raster = _raster(np.arange(5, dtype=float).reshape(1, 5))
    result = raster.cosample(raster, mask=np.array([[True, False, True, False, True]]))

    assert np.array_equal(result.self_values, [0, 2, 4])


@pytest.mark.parametrize("accessor", [False, True])
def test_point_cosample_uses_selected_band_validity(accessor: bool) -> None:
    """A point sample of a multiband raster must use validity from only the requested band."""

    # Place gaps at different points in each band to detect a mistaken validity band
    data = np.arange(30, dtype=float).reshape(5, 6)
    bands = np.stack((data, data + 100))
    bands[0, 1, 1] = np.nan
    bands[1, 3, 3] = np.nan
    raster: gu.Raster | gu.RasterAccessor = _raster(bands)
    x, y = raster.ij2xy(np.array([1, 2, 3]), np.array([1, 2, 3]))
    points = gu.PointCloud.from_xyz(x, y, np.array([1.0, 2.0, 3.0]), crs=raster.crs)
    if accessor:
        raster = gu.RasterAccessor.from_array(bands, raster.transform, raster.crs, nodata=-99999).rst

    result = raster.cosample(points, band=2, interpolation="nearest")

    assert np.array_equal(result.indices[0], [0, 1])
    assert np.array_equal(result.self_values, [107, 114])
    assert np.array_equal(result.other_values, [1, 2])
