"""Tests for cosampling at the same support on raster grids and point geometries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig


def _raster(data: NDArrayNum, *, x_origin: float = 0) -> gu.Raster:
    """Simplify creating a raster for tests below, to only need a one line call."""

    return gu.Raster.from_array(data, transform=from_origin(x_origin, data.shape[-2], 1, 1), crs=32633, nodata=-99999)


class TestCosample:
    """
    Checks cosample() calls that combine raster and point data, with eager inputs.

    See TestCosampleChunked further below for Dask/Multiprocessing tests.

    Here, we test that:
    - Point values can be adequately placed on a raster grid common support (with gridding).
    - Raster values can be adequately placed on a point locations common support (with interpolation).
    - Inputs with different CRS or nodata definition (Xarray with NaNs versus masked arrays) behaves properly during
        cosampling.
    """

    @pytest.mark.parametrize("caller", ["raster", "points"])
    @pytest.mark.parametrize("accessor", [False, True])
    @pytest.mark.parametrize("explicit_at", [False, True])
    def test_cosample__grid_points(self, caller: str, accessor: bool, explicit_at: bool) -> None:
        """Checks that gridding some points on a common raster support behaves as expected."""

        # Place two points around each grid pixel so their circular mean is known
        expected = np.arange(20, dtype=float).reshape(4, 5)
        raster = _raster(expected + 100)
        rows, columns = np.indices(raster.shape)
        x, y = raster.ij2xy(rows.ravel(), columns.ravel())
        point_values = np.repeat(expected.ravel(), 2) + np.tile([-2, 2], expected.size)
        points = gu.PointCloud.from_xyz(
            np.repeat(x, 2) + np.tile([-0.2, 0.2], expected.size),
            np.repeat(y, 2),
            point_values,
            crs=raster.crs,
        )
        keep = np.ones(raster.shape, dtype=bool)
        keep[1, 2] = False

        # Select raster output through either the conversion mode or an explicit grid
        first, second = (raster, points) if caller == "raster" else (points, raster)
        source = first
        if accessor:
            source = first.to_xarray().rst if caller == "raster" else first.ds.pc
            second = points.ds if caller == "raster" else raster.to_xarray()
        target_grid = raster.to_xarray() if accessor else raster
        target = {"at": target_grid} if explicit_at else {"raster_point_mode": "grid_points"}
        result = source.cosample(
            second,
            **target,
            grid_method="mean",
            grid_kwargs={"dist_nodata_pixel": 0.4, "min_points": 2},
            auxiliary={"offset": point_values + 10},
            auxiliary_at="other" if caller == "raster" else "self",
            mask=keep,
        )

        # Check the means, added point values, and common mask against arrays
        assert isinstance(result, xr.DataArray if accessor else gu.Raster)
        output = result.rst if accessor else result
        data = result.values if accessor else result.data.filled(np.nan)
        expected_bands = [expected + 100, expected] if caller == "raster" else [expected, expected + 100]
        expected_bands.append(expected + 10)
        assert output.transform == raster.transform
        np.testing.assert_allclose(data, np.where(keep, np.stack(expected_bands), np.nan))
        np.testing.assert_array_equal(points.data, point_values)

    @pytest.mark.parametrize("caller", ["raster", "points"])
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_cosample__resample_raster(self, caller: str, method: str) -> None:
        """Checks that raster resampling properly interpolates at irregular point coordinates."""

        # Create a ramp raster whose bilinear values can be calculated exactly
        rows, columns = np.indices((7, 9))
        raster = _raster((10 * rows + 2 * columns).astype(float))
        target_rows = np.array([1.2, 2.3, 4.1])
        target_columns = np.array([1.1, 3.2, 5.4])
        x, y = raster.ij2xy(target_rows, target_columns)
        points = gu.PointCloud.from_xyz(x, y, np.array([5.0, 6.0, 7.0]), crs=raster.crs)

        # Read the raster at point locations through either object
        first, second = (raster, points) if caller == "raster" else (points, raster)
        result = first.cosample(second, raster_point_mode="resample_raster", resample_method=method)

        # Check linear values on the slope and nearest values from the closest pixels
        expected = 10 * target_rows + 2 * target_columns
        if method == "nearest":
            expected = 10 * np.rint(target_rows) + 2 * np.rint(target_columns)
        raster_column = "self" if caller == "raster" else "other"
        np.testing.assert_allclose(result.ds[raster_column], expected)
        assert result.ds.geometry.equals(points.ds.geometry)

    def test_cosample__single_point_auxiliary_on_raster(self) -> None:
        """Checks edge case of a single point auxiliary input."""

        # Place one observation at the only pixel center so every output band has one known value
        raster = _raster(np.array([[10.0]]))
        x, y = raster.ij2xy(np.array([0]), np.array([0]))
        points = gu.PointCloud.from_xyz(x, y, np.array([2.0]), crs=raster.crs)
        auxiliary = np.array([7.0])

        # Grid point value and auxiliary onto the explicitly selected raster
        result = points.cosample(
            raster, auxiliary={"extra": auxiliary}, auxiliary_at="self", at=raster, grid_method="nearest"
        )

        # Check we still have same spatial dimensions and the original point value
        expected = np.array([[[2.0]], [[10.0]], [[7.0]]])
        np.testing.assert_array_equal(result.data.filled(np.nan), expected)
        np.testing.assert_array_equal(points.data, [2.0])

    def test_cosample__grid_crs_alignment(self) -> None:
        """Checks that gridding reprojects points from a different CRS when explicitly requested."""

        # Reproject points from the raster into a different CRS
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        points = raster.to_pointcloud().reproject(crs=4326)

        # Allow the points to move back onto the raster grid
        result = raster.cosample(points, raster_point_mode="grid_points", grid_method="nearest", align="reproject")

        # Check exact equality after cosample reprojection back to original CRS
        np.testing.assert_allclose(result.data[0], result.data[1])

    @pytest.mark.parametrize("accessor", [False, True])
    def test_cosample__different_point_locations_share_grid(self, accessor: bool) -> None:
        """
        Checks that point clouds with different X/Y coordinates can share a common support after gridding
        (functionality mostly useful for dense point cloud inputs).
        """

        # Place one observation near each grid center in two point clouds with deliberately different X coordinates,
        # but all within nearest neighbour range of the initial grid center
        values = np.arange(20, dtype=float).reshape(4, 5)
        raster = _raster(values)
        rows, columns = np.indices(raster.shape)
        x, y = raster.ij2xy(rows.ravel(), columns.ravel())
        first = gu.PointCloud.from_xyz(x, y, values.ravel(), crs=raster.crs)
        second = gu.PointCloud.from_xyz(x + 0.1, y, 2 * values.ravel(), crs=raster.crs)

        # Grid each point set onto the same support and return every input with the caller's container type
        source = first.ds.pc if accessor else first
        other = second.ds if accessor else second
        support = raster.to_xarray() if accessor else raster
        result = source.cosample(other, at=support, grid_method="nearest")

        # Check all values are almost equal, as the small coordinate offset makes each observation nearest to its
        # original grid pixel
        output = result.values if accessor else result.data.filled(np.nan)
        np.testing.assert_allclose(output, np.stack((values, 2 * values)))

    def test_cosample__resampling_nodata_options(self) -> None:
        """Checks nodata options passed through resampling kwargs."""

        # Put one missing pixel beside a point whose other interpolation neighbors equal one
        values = np.ones((5, 6), dtype=float)
        values[2, 2] = np.nan
        raster = _raster(values)
        x, y = raster.ij2xy(np.array([1.25, 3.0]), np.array([1.25, 3.0]))
        points = gu.PointCloud.from_xyz(x, y, np.array([10.0, 20.0]), crs=raster.crs)

        # Read the same points with strict and lenient missing data settings
        ignored = raster.cosample(points, resample_kwargs={"nodata_propagation": "ignore"})
        propagated = raster.cosample(points, resample_kwargs={"nodata_propagation": "propagate"})

        # Check that only the strict setting drops the point beside the missing pixel
        assert list(ignored.ds.index) == [0, 1]
        assert list(propagated.ds.index) == [1]
        np.testing.assert_allclose(ignored.ds["self"], 1)
        np.testing.assert_allclose(propagated.ds["self"], 1)


class TestRasterCosampleSupport:
    """
    Test module for common validity when cosample() returns a raster.

    Specifically, we test behaviour for input masks, selected bands and output shapes.
    """

    @pytest.mark.parametrize("accessor", [False, True])
    @pytest.mark.parametrize("input_type", ["raster", "numpy", "xarray"])
    @pytest.mark.parametrize("auxiliary_at", ["self", "other"])
    def test_cosample__common_validity_and_auxiliary(self, accessor: bool, input_type: str, auxiliary_at: str) -> None:
        """Checks that input values and the user mask properly determines the valid output pixels."""

        # Create inputs and the user mask with different pixels to exclude
        first = np.arange(20, dtype=float).reshape(4, 5)
        second, auxiliary = 10 * first, 100 * first
        first[0, 0], second[1, 1], auxiliary[2, 2] = np.nan, np.nan, np.nan
        mask = np.ones(first.shape, dtype=bool)
        mask[3, 3] = False
        raster = _raster(first)

        # Run the same public call through a Raster and an Xarray accessor
        source = raster.to_xarray().rst if accessor else raster
        other: Any = _raster(second)
        added: Any = auxiliary
        selected_mask: Any = mask
        if input_type == "numpy":
            other = second
        elif input_type == "xarray":
            # Plain DataArrays carry dimensions but inherit their coordinates from the geospatial input
            other = xr.DataArray(second, dims=("y", "x"))
            added = xr.DataArray(auxiliary, dims=("y", "x"))
            selected_mask = xr.DataArray(mask, dims=("y", "x"))
        elif accessor:
            other = other.to_xarray()
        result = source.cosample(other, auxiliary={"aux": added}, auxiliary_at=auxiliary_at, mask=selected_mask)
        assert isinstance(result, xr.DataArray if accessor else gu.Raster)
        output = result.rst if accessor else result
        data = result.to_numpy() if accessor else result.data.filled(np.nan)

        # Check the common support grid, validity mask, and band order
        assert output.shape == raster.shape
        assert output.transform == raster.transform
        assert output.crs == raster.crs
        assert output.tags["long_name"] == ("self", "other", "aux")
        expected = mask & np.isfinite(first) & np.isfinite(second) & np.isfinite(auxiliary)
        assert np.array_equal(np.isfinite(data), np.broadcast_to(expected, data.shape))

        # Check every selected value directly, including valid zeros
        assert np.array_equal(data[0, expected], first[expected])
        assert np.array_equal(data[1, expected], second[expected])
        assert np.array_equal(data[2, expected], auxiliary[expected])

    @pytest.mark.parametrize("at", ["self", "other", "explicit"])
    def test_cosample__raster_support_and_alignment(self, at: str) -> None:
        """Checks that cosample() uses the selected raster grid and reprojects only when allowed by user input."""

        # Shift the second raster by one pixel so the grids overlap but do not match
        first = _raster(np.arange(12, dtype=float).reshape(3, 4))
        second = _raster(np.arange(12, dtype=float).reshape(3, 4), x_origin=1)
        common_support = first if at == "self" else second
        selected_at = second if at == "explicit" else at

        # Allow reprojection onto the selected grid
        result = first.cosample(second, at=selected_at, align="reproject")

        # Check the chosen grid and the pixels outside the overlap
        assert isinstance(result, gu.Raster)
        assert result.transform == common_support.transform
        assert result.shape == common_support.shape
        assert result.count == 2
        assert np.count_nonzero(~np.ma.getmaskarray(result.data[0])) == 9

    def test_cosample__selected_bands_and_auxiliary_sources(self) -> None:
        """Checks that selected bands and auxiliary arrays use their specified input grid."""

        # Give each requested band distinct values so a wrong band is easy to detect
        base = np.arange(20, dtype=float).reshape(4, 5)
        first = _raster(np.stack((base, base + 100)))
        second = _raster(np.stack((2 * base, 2 * base + 200)))

        # Specify whether each auxiliary array shares the first or second raster grid
        result = first.cosample(
            second,
            band=2,
            other_band=2,
            auxiliary={"first_aux": base + 1, "second_aux": 2 * base + 2},
            auxiliary_at={"first_aux": "self", "second_aux": "other"},
        )

        # Check the band order and values through the public Raster API
        bands = result.split_bands()
        expected = (base + 100, 2 * base + 200, base + 1, 2 * base + 2)
        for band, values in zip(bands, expected):
            assert np.array_equal(band.data, values)
        assert result.tags["long_name"] == ("self", "other", "first_aux", "second_aux")

    @pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
    @pytest.mark.parametrize("accessor", [False, True])
    def test_cosample__singleton_spatial_dimensions(self, shape: tuple[int, int], accessor: bool) -> None:
        """Checks that one row or column remains a spatial dimension in a combined multiband raster."""

        # Select a few pixels from a grid with only one row or one column
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        raster = _raster(values)
        source = gu.RasterAccessor.from_array(values, raster.transform, raster.crs).rst if accessor else raster
        mask = values % 2 == 0
        result = source.cosample(source, mask=mask)

        # Check that both spatial dimensions remain separate from the two bands
        data = result.to_numpy() if accessor else result.data.filled(np.nan)
        assert data.shape == (2, *shape)
        assert np.array_equal(np.isfinite(data[0]), mask)
        assert np.array_equal(data[0, mask], values[mask])


class TestPointCosampleSupport:
    """
    This test module checks common support when cosample() returns points.

    It verifies various user inputs, masks, labels and sampling.
    """

    @pytest.mark.parametrize("caller", ["raster", "pointcloud"])
    @pytest.mark.parametrize("accessor", [False, True])
    def test_cosample__mixed_raster_and_point_inputs(self, caller: str, accessor: bool) -> None:
        """Checks that raster and point values are returned together at the point locations."""

        # Give the raster and point cloud a nodata value at different locations
        values = np.arange(30, dtype=float).reshape(5, 6)
        values[1, 2] = np.nan
        raster = _raster(values)
        rows, columns = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
        x, y = raster.ij2xy(rows, columns)
        points = gu.PointCloud.from_xyz(x, y, np.array([4.0, 5.0, 6.0, np.nan]), crs=raster.crs)
        points.ds.index = ["a", "b", "c", "d"]
        source_column, source_bounds = points.data_column, points.bounds

        # Use the point locations as common support through either spatial object or accessor
        source, other = (raster, points) if caller == "raster" else (points, raster)
        if accessor:
            source = source.to_xarray().rst if caller == "raster" else source.ds.pc
            other = points.ds if caller == "raster" else raster.to_xarray()
        result = source.cosample(other, resample_method="nearest")
        assert isinstance(result, gpd.GeoDataFrame if accessor else gu.PointCloud)
        output = result if accessor else result.ds
        output_points = result.pc if accessor else result

        # Check the original point labels, geometry, and two named value columns
        assert np.array_equal(output.index, ["a", "c"])
        assert output.geometry.equals(points.ds.geometry.iloc[[0, 2]])
        assert output_points.data_column == "self"
        raster_values = values[rows[[0, 2]], columns[[0, 2]]]
        expected = (raster_values, [4.0, 6.0]) if caller == "raster" else ([4.0, 6.0], raster_values)
        assert np.array_equal(output["self"], expected[0])
        assert np.array_equal(output["other"], expected[1])

        # Check that point count and bounds describe the selected points while the original metadata stays unchanged
        assert output_points.point_count == 2
        assert np.array_equal(output_points.bounds, points.ds.iloc[[0, 2]].total_bounds)
        assert points.point_count == 4
        assert points.data_column == source_column
        assert points.bounds == source_bounds

    @pytest.mark.parametrize("at", [None, "self", "other", "explicit", "raw_self", "raw_other"])
    @pytest.mark.parametrize("auxiliary_type", ["numpy", "xarray"])
    def test_cosample__point_auxiliaries_and_labels(self, at: str | None, auxiliary_type: str) -> None:
        """Checks that point labels, geometry and auxiliary columns are unchanged in the result."""

        # Create 3D points with duplicate row labels
        positions = np.arange(8, dtype=float)
        first = gu.PointCloud.from_xyz(positions, positions**2, positions, crs=32633, use_z=True)
        second = gu.PointCloud.from_xyz(positions, positions**2, 2 * positions, crs=32633, use_z=True)
        first.ds.index = ["a", "b", "a", "c", "d", "e", "f", "g"]
        second.ds.index = np.arange(8) + 10
        auxiliary = 3 * positions
        auxiliary[3] = np.nan

        # Choose either point input as the common support
        selected_at = second if at == "explicit" else at
        other: Any = second
        added: Any = xr.DataArray(auxiliary, dims="point") if auxiliary_type == "xarray" else auxiliary
        auxiliary_at = "self"
        if at in {"raw_self", "raw_other"}:
            # A plain second array uses the first point locations even when auxiliary_at selects the other input
            other = second.data
            selected_at = "other" if at == "raw_other" else None
            auxiliary_at = "other" if at == "raw_other" else "self"
        result = first.cosample(other, auxiliary={"weight": added}, auxiliary_at=auxiliary_at, at=selected_at)
        support = second if at in {"other", "explicit"} else first
        expected = np.array([0, 1, 2, 4, 5, 6, 7])

        # Check the chosen labels, 3D geometry, values, and column order
        assert result.ds.geometry.equals(support.ds.geometry.iloc[expected])
        assert list(result.ds.columns) == ["self", "other", "weight", "geometry"]
        assert np.array_equal(result.ds["other"], 2 * result.ds["self"])
        assert np.array_equal(result.ds["weight"], 3 * result.ds["self"])

    @pytest.mark.parametrize("accessor", [False, True])
    @pytest.mark.parametrize("common_support", ["raster", "points"])
    def test_cosample__auxiliary_point_column(self, accessor: bool, common_support: str) -> None:
        """Checks that a named point column can supply auxiliary values for raster or point output."""

        # Give each point an active data column and a separate auxiliary column at a known raster pixel
        values = np.arange(20, dtype=float).reshape(4, 5)
        raster = _raster(values)
        rows, columns = np.indices(raster.shape)
        x, y = raster.ij2xy(rows.ravel(), columns.ravel())
        points = gu.PointCloud.from_xyz(x, y, values.ravel(), crs=raster.crs)
        weights = 3 * values.ravel() + 100
        points.ds["weight"] = weights
        original_column = points.data_column

        # Select the auxiliary column without changing the active values of either primary input
        source = points.ds.pc if accessor else points
        other = points.ds if accessor else points
        support = raster.to_xarray() if accessor else raster
        selected_at = support if common_support == "raster" else "self"
        result = source.cosample(other, auxiliary={"chosen": (other, "weight")}, at=selected_at, grid_method="nearest")

        # Check the selected column against its own values and preserve the original active column
        if common_support == "raster":
            output = result.values if accessor else result.data.filled(np.nan)
            assert np.array_equal(output, np.stack((values, values, weights.reshape(values.shape))))
        else:
            output = result if accessor else result.ds
            assert np.array_equal(output["chosen"], weights)
            assert np.array_equal(output["self"], values.ravel())
            assert output.geometry.equals(points.ds.geometry)
        assert points.data_column == original_column
        assert np.array_equal(points.data, values.ravel())

    def test_cosample__plain_auxiliaries_at_reprojected_points(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that plain auxiliary arrays follow their point cloud through reprojection."""

        # Choose the output coordinates by projecting the original points once, avoiding roundtrip rounding differences
        values = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(10 + values / 100, 45 + values / 100, values, crs=4326)
        support = points.reproject(crs=32632)
        auxiliaries = {"scaled": 3 * values, "offset": values + 100}

        # Count coordinate comparisons to check that arrays tied to the same point cloud share one comparison
        comparisons: list[tuple[Any, Any]] = []
        original_comparison = gu.PointCloud.georeferenced_coords_equal

        def record_comparison(first: gu.PointCloud, second: Any) -> bool:
            """Count coordinate comparisons while preserving their ordinary result."""

            comparisons.append((first, second))
            return original_comparison(first, second)

        monkeypatch.setattr(gu.PointCloud, "georeferenced_coords_equal", record_comparison)
        result = points.cosample(2 * values, auxiliary=auxiliaries, auxiliary_at="self", at=support, align="reproject")

        # Check all original values at the projected coordinates after comparing their source point cloud once
        assert len(comparisons) == 1
        expected = np.column_stack((values, 2 * values, 3 * values, values + 100))
        assert np.array_equal(result.ds[["self", "other", "scaled", "offset"]], expected)
        assert result.ds.geometry.equals(support.ds.geometry)
        assert points.crs.to_epsg() == 4326

    @pytest.mark.parametrize("mask_type,mask_mode", [("vector", "inside"), ("vector", "outside"), ("raster", "inside")])
    def test_cosample__mask_matches_statistics(self, mask_type: str, mask_mode: str) -> None:
        """Checks that cosample() and stats() apply spatial masks to the same points."""

        # 1/ Prepare points and equivalent spatial masks
        # Place two points in each half of a raster, with distinct values and no nodata values
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        rows, columns = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
        x, y = raster.ij2xy(rows, columns)
        mask: gu.Vector | gu.Raster
        if mask_type == "vector":
            # Put the fifth point exactly on the polygon's right edge, where create_mask() must exclude it
            x, y = np.append(x, 3.0), np.append(y, 2.0)
            # Extend the polygon above the raster so the first point lies inside rather than on its boundary
            mask = gu.Vector(gpd.GeoDataFrame({"zone": ["west"]}, geometry=[box(0, 0, 3, 6)], crs=raster.crs))
        else:
            keep = np.indices(raster.shape)[1] < 3
            mask = gu.Raster.from_array(keep, raster.transform, raster.crs)
            # A point beyond the grid must receive a nodata mask value and remain excluded
            x, y = np.append(x, 100.0), np.append(y, 100.0)
        points = gu.PointCloud.from_xyz(x, y, np.arange(1, len(x) + 1, dtype=float) * 10, crs=raster.crs)

        # 2/ Apply the same mask through each public operation
        # Compare point cosampling, one summary, and a single group with its complete group mask
        sampled = points.cosample(points, mask=mask, mask_mode=mask_mode)
        summary = points.stats(["mean", "validinliercount"], mask=mask, mask_mode=mask_mode)
        grouped, masks = points.stats(
            "mean",
            values={"height": None},
            by={"zone": np.zeros(len(x), dtype=int)},
            categories={"zone": [0]},
            mask=mask,
            mask_mode=mask_mode,
            return_masks=True,
        )

        # 3/ Check the selected point positions and values
        # Follow create_mask(): polygon boundaries belong to the outside selection
        if isinstance(mask, gu.Vector):
            inside = np.asarray(mask.create_mask(ref=points, as_array=True), dtype=bool)
            assert np.array_equal(inside, [True, True, False, False, False])
            expected = np.flatnonzero(inside if mask_mode == "inside" else ~inside)
        else:
            expected = np.array([0, 1])
        expected_mean = np.asarray(points.data)[expected].mean()
        assert np.array_equal(sampled.ds.index, expected)
        assert np.array_equal(np.flatnonzero(masks[0].data), expected)
        assert summary == pytest.approx({"mean": expected_mean, "validinliercount": len(expected)})
        np.testing.assert_allclose(grouped["height"], [[len(expected), expected_mean]])

        # Categorical vector zones use intersections and therefore include the boundary point
        if isinstance(mask, gu.Vector) and mask_mode == "inside":
            zonal = points.stats("mean", values={"height": None}, by={"zone": (mask, "zone")})
            np.testing.assert_allclose(zonal["height"], [[3, (10 + 20 + 50) / 3]])

    def test_cosample__masked_auxiliary_and_mask(self) -> None:
        """Checks that missing added point values and mask values remove their matching rows."""

        # Exclude different points through a missing added value, a missing mask value, and a false mask value
        positions = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, positions, crs=32633)
        auxiliary = np.ma.array(np.arange(5), mask=[False, True, False, False, False])
        mask = np.ma.array([True, True, True, False, True], mask=[False, False, True, False, False])
        result = points.cosample(points, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=mask)

        # Check that only the first and last points remain in their original order
        assert np.array_equal(result.ds.index, [0, 4])
        assert np.array_equal(result.ds["aux"], [0, 4])

    @pytest.mark.parametrize("singleton_band", [False, True])
    def test_cosample__raster_auxiliary_interpolation_options(self, singleton_band: bool) -> None:
        """Checks that a plain raster auxiliary uses the requested interpolation options."""

        # Place points around a nodata auxiliary pixel while leaving both primary inputs fully valid
        raster = _raster(np.arange(99, dtype=float).reshape(9, 11))
        auxiliary = raster.data.filled(np.nan) + 1000
        auxiliary[4, 5] = np.nan
        # The primary raster's nodata sentinel is a valid number in this independent auxiliary array
        auxiliary[1, 1] = -99999
        positions = np.arange(40)
        x, y = raster.ij2xy(1 + positions % 7, 1 + positions % 9)
        points = gu.PointCloud.from_xyz(x, y, positions.astype(float), crs=raster.crs)

        # Use Raster.interp_points() to identify which points remain valid around the nodata auxiliary pixel
        auxiliary_raster = gu.Raster.from_array(
            np.ma.masked_invalid(auxiliary), raster.transform, raster.crs, nodata=None
        )
        expected = auxiliary_raster.interp_points((x, y), method="nearest", as_array=True, dist_nodata_spread=2)
        kept = np.flatnonzero(np.isfinite(expected))
        raw_auxiliary = auxiliary[np.newaxis, ...] if singleton_band else auxiliary
        result = points.cosample(
            raster,
            auxiliary={"offset": raw_auxiliary},
            auxiliary_at="other",
            resample_method="nearest",
            resample_kwargs={"dist_nodata_spread": 2},
        )

        # Check the exact finite positions returned by interpolation for two-dimensional and one-band inputs
        assert 0 < kept.size < positions.size
        assert -99999 in expected[kept]
        assert np.array_equal(result.ds.index, kept)
        assert np.array_equal(result.ds["offset"], expected[kept])

    @pytest.mark.parametrize("accessor", [False, True])
    def test_cosample__selected_band_validity(self, accessor: bool) -> None:
        """Checks that only nodata pixels in the requested raster band remove points."""

        # Put nodata pixels at different point locations in the two raster bands
        data = np.arange(30, dtype=float).reshape(5, 6)
        bands = np.stack((data, data + 100))
        bands[0, 1, 1], bands[1, 3, 3] = np.nan, np.nan
        raster = _raster(bands)
        x, y = raster.ij2xy(np.array([1, 2, 3]), np.array([1, 2, 3]))
        points = gu.PointCloud.from_xyz(x, y, np.array([1.0, 2.0, 3.0]), crs=raster.crs)

        # Request only the second band through a Raster and an Xarray accessor
        source = raster.to_xarray().rst if accessor else raster
        other = points.ds if accessor else points
        result = source.cosample(other, band=2, resample_method="nearest")
        output = result if accessor else result.ds

        # Check that a point over nodata in only the unused first band remains
        assert np.array_equal(output.index, [0, 1])
        assert np.array_equal(output["self"], [107, 114])
        assert np.array_equal(output["other"], [1, 2])

    def test_cosample__subsampling_and_independent_result(self) -> None:
        """Checks that subsampling returns points in source order and data independent of its inputs."""

        # Use duplicate row labels so sampling must follow row positions
        positions = np.arange(20, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, positions, crs=32633)
        points.ds.index = np.tile(["z", "a"], 10)
        original = points.ds.copy(deep=True)
        result = points.cosample(points, subsample=5, random_state=42)
        repeated = points.cosample(points, subsample=5, random_state=42)

        # Check repeatable selection in the original point order
        assert len(result.ds) == 5
        assert np.all(np.diff(result.ds["self"]) > 0)
        assert_geodataframe_equal(result.ds, repeated.ds)

        # Check that changing the returned PointCloud does not change either input
        result.ds["self"] = -1
        assert_geodataframe_equal(points.ds, original)


class TestCosampleChunked:
    """Test module for cosample() with Dask and Multiproc backends: loading behavior and exact equality with eager."""

    @pytest.mark.parametrize("auxiliary_type", ["numpy", "dask", "column"])
    def test_cosample__dask_point_auxiliaries_on_raster(self, auxiliary_type: str) -> None:
        """Checks that lazy point gridding assigns array or column auxiliaries to the correct raster pixels."""

        import_optional("dask")
        import dask.array as da

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # Give every pixel one point and a distinct auxiliary value, using duplicate labels to expose index alignment
        _register_dask_pointcloud_accessor()
        values = np.arange(12, dtype=float).reshape(3, 4)
        raster = _raster(values + 100)
        rows, columns = np.indices(raster.shape)
        x, y = raster.ij2xy(rows.ravel(), columns.ravel())
        points = gu.PointCloud.from_xyz(x, y, values.ravel(), crs=raster.crs)
        auxiliary_values = 3 * values.ravel() + 7
        points.ds["weight"] = auxiliary_values
        points.ds.index = np.tile(["a", "b"], 6)

        # Partition points independently from Dask arrays, or select the existing point column
        lazy_points = dgpd.from_geopandas(points.ds, npartitions=3, sort=False)
        lazy_points.pc.data_column = points.data_column
        target = raster.to_xarray().chunk({"y": 2, "x": 3})
        auxiliary: Any = auxiliary_values
        if auxiliary_type == "dask":
            auxiliary = da.from_array(auxiliary_values, chunks=5)
        elif auxiliary_type == "column":
            auxiliary = (lazy_points, "weight")
        result = lazy_points.pc.cosample(
            target, auxiliary={"extra": auxiliary}, auxiliary_at="self", at=target, grid_method="nearest"
        )

        # Calculate the same result from the eager point cloud and raster
        eager_auxiliary: Any = (points, "weight") if auxiliary_type == "column" else auxiliary_values
        expected = points.cosample(
            raster,
            auxiliary={"extra": eager_auxiliary},
            auxiliary_at="self",
            at=raster,
            grid_method="nearest",
        )

        # Check that neither input nor the result is computed until their values are requested
        assert target.chunks is not None and result.chunks is not None
        assert not lazy_points.pc.is_loaded
        assert lazy_points.pc.data_column == points.data_column
        assert np.array_equal(result.compute().values, expected.data.filled(np.nan), equal_nan=True)
        assert not lazy_points.pc.is_loaded

    @pytest.mark.parametrize("chunks", [(7, 11), (32, 47), (256, 256)])
    @pytest.mark.parametrize("caller", ["raster", "points"])
    def test_cosample__dask_gridding_chunks(self, chunks: tuple[int, int], caller: str, tmp_path: Path) -> None:
        """Checks that Dask point gridding returns the same seeded sample as eager gridding for each chunk size."""

        # Place one point at each grid pixel so the nearest-neighbor gridding is exact
        pytest.importorskip("dask.array")
        pytest.importorskip("dask_geopandas")
        values = np.arange(65 * 97, dtype=float).reshape(65, 97)
        raster = _raster(values)
        points = raster.to_pointcloud()
        lazy_raster = raster.to_xarray().chunk({"y": chunks[0], "x": chunks[1]})

        # Write the points and reopen them as lazy partitions
        point_file = tmp_path / "observations.gpkg"
        points.to_file(point_file)
        lazy_points = gu.open_pointcloud(str(point_file), data_column=points.data_column, chunks=1400)

        # Grid lazy point data with two partition sizes and both public calling objects
        source, other = (lazy_raster.rst, lazy_points) if caller == "raster" else (lazy_points.pc, lazy_raster)
        result = source.cosample(
            other,
            raster_point_mode="grid_points",
            grid_method="nearest",
            grid_kwargs={"chunksizes": chunks},
            subsample=200,
            random_state=42,
            strategy="topk",
        )
        expected = raster.cosample(raster, subsample=200, random_state=42, strategy="topk")

        # Check that all runs stay lazy and select the same pixels and values as the eager call
        assert result.data.chunks is not None
        assert not lazy_points.pc.is_loaded
        assert np.array_equal(result.compute().values, expected.data.filled(np.nan), equal_nan=True)
        assert not lazy_points.pc.is_loaded

    @pytest.mark.parametrize("reproject", [False, True])
    def test_cosample__multiproc_point_columns_loading(self, tmp_path: Path, reproject: bool) -> None:
        """Checks that aligning and gridding point columns doesn't load their file."""

        # Store one point at every grid center so both selected columns have exact expected raster values
        values = np.arange(30, dtype=float).reshape(5, 6)
        raster = _raster(values + 100)
        points = _raster(values + 10).to_pointcloud()
        points.ds["weight"] = 3 * values.ravel() + 7
        if reproject:
            points = points.reproject(crs=4326)
        expected = raster.cosample(
            points,
            auxiliary={"weight": (points, "weight")},
            at=raster,
            grid_method="nearest",
            align="reproject",
        )
        point_file = tmp_path / "observations.gpkg"
        points.to_file(point_file)
        unloaded = gu.PointCloud(point_file, data_column=points.data_column)
        assert not unloaded.is_loaded

        # Read both point columns into raster tiles without copying or loading the point source in the parent
        outfile = tmp_path / "cosampled.tif"
        result = raster.cosample(
            unloaded,
            auxiliary={"weight": (unloaded, "weight")},
            at=raster,
            grid_method="nearest",
            align="reproject",
            mp_config=MultiprocConfig(chunks=(2, 3), outfile=str(outfile)),
        )

        # Check that the original active column is unchanged, loading behaviour and that the result matches eager
        assert not unloaded.is_loaded
        assert unloaded.data_column == points.data_column
        assert not result.is_loaded
        assert outfile.exists()
        assert np.array_equal(result.data.filled(np.nan), expected.data.filled(np.nan), equal_nan=True)
        assert not unloaded.is_loaded

    def test_cosample__multiproc_preserves_point_locations(self, tmp_path: Path) -> None:
        """Checks that Multiproc alignment returns exact point locations for values and plain auxiliaries."""

        # Project irregular geographic coordinates so rounding them through LAS would change their exact locations
        positions = np.arange(11, dtype=float)
        points = gu.PointCloud.from_xyz(10 + positions / 31, 45 + positions / 47, positions + 1, crs=4326)
        support = points.reproject(crs=32632)
        point_file = tmp_path / "geographic.gpkg"
        points.to_file(point_file)
        unloaded = gu.PointCloud(point_file, data_column=points.data_column)

        # Pass rectangular chunks and a raster output path to check point output leaves both options unchanged
        config = MultiprocConfig(chunks=(2, 3), outfile=str(tmp_path / "unused-output.tif"))

        # Compare eager and Multiproc alignment of unloaded point values and two arrays at reprojected point locations
        expected = points.cosample(
            2 * positions,
            auxiliary={"offset": positions + 10},
            auxiliary_at="self",
            at=support,
            align="reproject",
        )
        result = unloaded.cosample(
            2 * positions,
            auxiliary={"offset": positions + 10},
            auxiliary_at="self",
            at=support,
            align="reproject",
            mp_config=config,
        )

        # Point matching requires identical ordered X/Y values, no source load or config change is needed
        assert result.georeferenced_coords_equal(support)
        assert np.array_equal(result.ds[["self", "other", "offset"]], expected.ds[["self", "other", "offset"]])
        assert not unloaded.is_loaded
        assert config.chunks == (2, 3)
        assert config.driver is None
        assert not Path(config.outfile).exists()

    def test_cosample__multiproc_distinct_tempfiles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Checks that point gridding and shifted raster inputs use separate temporary files and preserve eager values.
        """

        # 1/ Prepare distinguishable values on a target grid and two shifted auxiliary grids
        # Different offsets expose an intermediate temporary raster files being overwritten by a later spatial operation
        rows, columns = np.indices((12, 15))
        values = (10 * rows + columns).astype(float)
        raster = _raster(values)
        points = _raster(values + 1000).to_pointcloud()
        auxiliaries = {"east": _raster(values + 2000, x_origin=1), "west": _raster(values + 3000, x_origin=-1)}
        raster_mask = gu.Raster.from_array(columns % 4 != 0, auxiliaries["west"].transform, raster.crs)
        options: dict[str, Any] = {"raster_point_mode": "grid_points", "grid_method": "nearest", "align": "reproject"}
        expected = raster.cosample(points, auxiliary=auxiliaries, mask=raster_mask, **options)

        # Reopen all raster inputs unloaded so every spatial stage must read its own source file
        unloaded = {}
        for name, source in {"reference": raster, **auxiliaries, "mask": raster_mask}.items():
            filename = tmp_path / f"{name}.tif"
            source.to_file(filename)
            unloaded[name] = gu.Raster(filename, load_data=False, is_mask=name == "mask")

        # 2/ Record temporary destinations while leaving their normal creation and cleanup unchanged
        intermediate_paths: list[Path] = []
        original_temporary = MultiprocConfig.temporary

        @contextmanager
        def record_temporary(config: MultiprocConfig) -> Iterator[MultiprocConfig]:
            """
            Record each intermediate filename without changing its normal context lifetime.
            """
            with original_temporary(config) as temporary:
                intermediate_paths.append(Path(temporary.outfile))
                yield temporary

        monkeypatch.setattr(MultiprocConfig, "temporary", record_temporary)
        outfile = tmp_path / "cosampled.tif"
        result = unloaded["reference"].cosample(
            points,
            auxiliary={"east": unloaded["east"], "west": unloaded["west"]},
            mask=unloaded["mask"],
            mp_config=MultiprocConfig(chunks=(5, 6), outfile=str(outfile)),
            **options,
        )

        # 3/ Check that the final file is still readable after all distinct intermediate files have been removed
        assert len(intermediate_paths) >= 4
        assert len(set(intermediate_paths)) == len(intermediate_paths)
        assert outfile not in intermediate_paths
        assert all(not path.exists() for path in intermediate_paths)
        assert outfile.exists()
        assert not result.is_loaded
        assert all(not source.is_loaded for source in unloaded.values())
        assert result.transform == raster.transform
        assert result.tags["long_name"] == ("self", "other", "east", "west")
        np.testing.assert_allclose(result.data.filled(np.nan), expected.data.filled(np.nan))

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("raster_mask", [False, True])
    def test_cosample__masked_integers_and_boolean_masks(self, lazy: bool, raster_mask: bool) -> None:
        """Checks that nodata and boolean masks do not convert valid integers."""

        # Place source nodata, masked auxiliary data, a masked user-mask pixel, and a false mask pixel separately
        data = np.ma.array(np.arange(30, dtype=np.int32).reshape(5, 6), mask=False)
        data.mask[0, 0] = True
        auxiliary = np.ma.array(2 * data.data, mask=False)
        auxiliary.mask[1, 1] = True
        mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
        mask.mask[2, 2], mask[3, 3] = True, False
        first = _raster(data)
        selected_mask = first.from_array(mask, first.transform, first.crs) if raster_mask else mask
        expected_output = first.cosample(
            first, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=selected_mask
        ).data.filled(np.nan)

        # Cosample eager or Dask rasters with the user mask supplied as an array or raster
        source = first
        if lazy:
            da = pytest.importorskip("dask.array")
            source = gu.RasterAccessor.from_array(
                da.from_array(data.astype(float).filled(np.nan), chunks=(2, 3)), first.transform, first.crs
            ).rst
            if raster_mask:
                # Convert the raster mask to Xarray with its masked pixel excluded and its values still boolean
                selected_mask = gu.RasterAccessor.from_array(
                    selected_mask.data.filled(False), selected_mask.transform, selected_mask.crs
                )
        result = source.cosample(source, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=selected_mask)
        output = result.to_numpy() if lazy else result.data.filled(np.nan)

        # Compare every output value with eager and check the common validity mask directly
        if lazy:
            assert result.chunks is not None
        assert np.array_equal(output, expected_output, equal_nan=True)
        expected = ~data.mask & ~auxiliary.mask & mask.filled(False)
        assert np.array_equal(np.isfinite(output[0]), expected)
        assert np.array_equal(output[0, expected], data.data[expected])
        assert np.array_equal(output[2, expected], auxiliary.data[expected])

    @pytest.mark.parametrize("subsample", [1, 12, 0.25])
    def test_cosample__dask_raster_chunks(self, subsample: int | float) -> None:
        """Checks that Dask output stays lazy and topk does not depend on chunk sizes."""

        # Create the two Dask rasters with different chunk layouts
        da = pytest.importorskip("dask.array")
        array = np.arange(63, dtype=float).reshape(7, 9)
        transform = from_origin(0, 7, 1, 1)
        first = gu.RasterAccessor.from_array(da.from_array(array, chunks=(2, 4)), transform, 32633)
        second = gu.RasterAccessor.from_array(da.from_array(2 * array, chunks=(4, 3)), transform, 32633)

        # Repeat the same seeded sample after changing one chunk layout
        result = first.rst.cosample(second, subsample=subsample, random_state=42, strategy="topk")
        changed = first.chunk({"y": 4, "x": 3}).rst.cosample(
            second, subsample=subsample, random_state=42, strategy="topk"
        )
        assert isinstance(result, xr.DataArray)
        assert isinstance(result.data, da.Array)
        assert isinstance(first.data, da.Array)

        # Check that the result remains Dask-backed and matches eager selected pixels and values
        eager = _raster(array).cosample(_raster(2 * array), subsample=subsample, random_state=42, strategy="topk")
        output = result.compute().to_numpy()
        assert np.array_equal(output, changed.to_numpy(), equal_nan=True)
        assert np.array_equal(output, eager.data.filled(np.nan), equal_nan=True)
        count = array.size if subsample == 1 else int(subsample * array.size) if subsample < 1 else subsample
        assert np.count_nonzero(np.isfinite(output[0])) == count
        np.testing.assert_allclose(output[1], 2 * output[0], equal_nan=True)

    @pytest.mark.parametrize(
        "mask_type,chunks,workers",
        [("array", (4, 5), False), ("raster", (5, 4), False), ("vector", (4, 5), True)],
    )
    def test_cosample__multiproc_raster_matches_eager_and_dask(
        self, mask_type: str, chunks: tuple[int, int], workers: bool, tmp_path: Path
    ) -> None:
        """
        Checks that Multiproc does not load raster inputs and matches eager and Dask results for each mask.
        """

        import_optional("dask")
        from geoutils.multiproc.cluster import MpCluster

        # 1/ Create selected bands, auxiliary values, and masks with nodata at different locations
        # Add unused first bands with large offsets so the result reveals if a worker reads the wrong band
        values = np.arange(180, dtype=float).reshape(12, 15)
        first = _raster(np.stack((values + 1000, values)))
        second = _raster(np.stack((values + 2000, 2 * values)))
        auxiliary = _raster(np.stack((values + 3000, 3 * values)))
        first.data[0, 1, 1] = np.ma.masked
        first.data[1, 2, 3] = np.ma.masked
        second.data[1, 4, 5] = np.ma.masked
        auxiliary.data[1, 6, 7] = np.ma.masked
        raw_auxiliary = 4 * values
        raw_auxiliary[8, 9] = np.nan

        # Use a regular boolean pattern for array/raster masks, and a polygon for the vector case
        keep = values % 7 != 0
        raster_mask = gu.Raster.from_array(keep, first.transform, first.crs)
        mask: Any = keep
        if mask_type == "raster":
            mask = raster_mask
        elif mask_type == "vector":
            mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 10, 12)], crs=first.crs))
        options: dict[str, Any] = {
            "band": 2,
            "other_band": 2,
            "auxiliary_at": {"raw": "self"},
            "subsample": 17,
            "random_state": 42,
            "strategy": "topk",
        }

        # 2/ Calculate eager and differently chunked Dask references through the public method
        eager = first.cosample(second, auxiliary={"scaled": (auxiliary, 2), "raw": raw_auxiliary}, mask=mask, **options)
        expected = eager.data.filled(np.nan)
        for rows, columns in ((4, 5), (5, 4)):
            lazy_first = first.to_xarray().chunk({"y": rows, "x": columns})
            lazy_second = second.to_xarray().chunk({"y": columns, "x": rows})
            lazy_auxiliary = auxiliary.to_xarray().chunk({"y": rows, "x": columns})
            lazy_mask = mask
            if mask_type == "raster":
                lazy_mask = gu.RasterAccessor.from_array(mask.data, mask.transform, mask.crs)
            lazy = lazy_first.rst.cosample(
                lazy_second,
                auxiliary={"scaled": (lazy_auxiliary, 2), "raw": raw_auxiliary},
                mask=lazy_mask,
                **options,
            )
            assert lazy.data.chunks is not None
            assert np.array_equal(lazy.compute().values, expected, equal_nan=True)

        # Write three raster files to disk with the selected bands and nodata patterns used by the eager reference
        unloaded = []
        for name, raster in (("first", first), ("second", second), ("auxiliary", auxiliary)):
            filename = tmp_path / f"{name}.tif"
            raster.to_file(filename)
            unloaded.append(gu.Raster(filename, load_data=False))
        mp_mask = mask
        if mask_type == "raster":
            # Write the boolean raster mask to disk so Multiproc reads it by tile
            mask_file = tmp_path / "mask.tif"
            raster_mask.to_file(mask_file)
            mp_mask = gu.Raster(mask_file, is_mask=True, load_data=False)

        # 3/ Write the cosampled raster to disk by tile, using worker processes for the vector-mask case
        outfile = tmp_path / "cosampled.tif"
        with MpCluster({"nb_workers": 2}) if workers else nullcontext(None) as cluster:
            configuration = MultiprocConfig(chunks=chunks, outfile=str(outfile), cluster=cluster)
            result = unloaded[0].cosample(
                unloaded[1],
                auxiliary={"scaled": (unloaded[2], 2), "raw": raw_auxiliary},
                mask=mp_mask,
                mp_config=configuration,
                **options,
            )
        assert outfile.exists()
        assert not result.is_loaded
        assert all(not source.is_loaded for source in unloaded)
        # Check that processing Multiproc tiles does not change the caller's original boolean mask
        assert np.array_equal(keep, values % 7 != 0)
        if mask_type == "raster":
            assert not mp_mask.is_loaded

        # Check complete output bands and the exact common sample after temporary worker files have been cleaned up
        assert np.array_equal(result.data.filled(np.nan), expected, equal_nan=True)
        assert np.count_nonzero(np.isfinite(expected[0])) == 17
        assert result.georeferenced_grid_equal(first)
        assert tuple(result.tags["long_name"]) == ("self", "other", "scaled", "raw")

    @pytest.mark.parametrize("lazy", [False, True])
    def test_cosample__no_replacement_after_interpolation(self, lazy: bool) -> None:
        """Checks that points rejected during interpolation are not replaced in the sample."""

        # Place points around one raster nodata pixel, including neighbors rejected by slinear's default nodata spread
        raster = _raster(np.ones((11, 11), dtype=float))
        raster.data[5, 5] = np.ma.masked
        rows, columns = np.meshgrid(np.arange(3, 8), np.arange(3, 8), indexing="ij")
        x, y = raster.ij2xy(rows.ravel(), columns.ravel())
        points = gu.PointCloud.from_xyz(x, y, np.arange(x.size, dtype=float), crs=raster.crs).ds
        options = {"subsample": 23, "random_state": 42}

        # Select almost every initially valid point so final interpolation removes some selected nodata neighbors
        coverage_values = raster.interp_points((x, y), method="slinear", dist_nodata_spread=0, as_array=True)
        candidate_validity = np.isfinite(coverage_values)
        selected = points.pc.cosample(points, mask=candidate_validity, **options)
        interpolated_values = raster.interp_points((x, y), method="slinear", as_array=True)
        selected_indices = selected.index.to_numpy()
        expected_indices = selected_indices[np.isfinite(interpolated_values[selected_indices])]
        expected = points.pc.cosample(raster.to_xarray(), resample_method="slinear", **options)
        assert candidate_validity.sum() == 24
        assert 0 < len(expected_indices) < options["subsample"]

        # Apply the same seeded selection with raster values read eagerly or through Dask
        source: Any = points
        other: Any = raster.to_xarray()
        if lazy:
            dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
            from geoutils.pointcloud.pd_accessor import (
                _register_dask_pointcloud_accessor,
            )

            _register_dask_pointcloud_accessor()
            source = dgpd.from_geopandas(points, npartitions=3, sort=False)
            other = raster.to_xarray().chunk({"y": 6, "x": 6})
        result = source.pc.cosample(other, resample_method="slinear", **options)
        if lazy:
            assert not source.pc.is_loaded
        output = result.compute() if lazy else result

        # Return the originally selected finite rows in their original order, with no replacement points
        assert_geodataframe_equal(output, expected)
        assert np.array_equal(output.index.to_numpy(), expected_indices)
        np.testing.assert_allclose(output["other"], interpolated_values[expected_indices])
        assert np.array_equal(output.geometry.to_numpy(), points.geometry.iloc[expected_indices].to_numpy())
        if lazy:
            assert not source.pc.is_loaded and not result.pc.is_loaded

    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_cosample__multiproc_raster_inputs_not_loaded(self, method: str, tmp_path: Path) -> None:
        """Checks that Multiproc point output matches eager while raster inputs stay unloaded."""

        # 1/ Prepare an affine raster, a boolean raster mask, and irregular point observations
        # Fractional pixel locations make nearest and linear interpolation produce distinct, predictable values
        rows, columns = np.indices((12, 15))
        values = (10 * rows + 2 * columns).astype(float)
        raster = _raster(np.stack((values + 1000, values)))
        raster.data[1, 4, 5] = np.ma.masked
        raster.data[0, 2, 3] = np.ma.masked
        keep = columns % 4 != 0
        raster_mask = gu.Raster.from_array(keep, raster.transform, raster.crs)
        positions = np.arange(24)
        x, y = raster.ij2xy(1.2 + positions % 8, 1.3 + positions % 11)
        points = gu.PointCloud.from_xyz(x, y, positions.astype(float), crs=raster.crs)
        points.ds.index = np.repeat(np.arange(12), 2)
        auxiliary = 3 * positions.astype(float)
        auxiliary[3] = np.nan

        # 2/ Calculate the eager reference from loaded inputs
        options: dict[str, Any] = {
            "other_band": 2,
            "auxiliary": {"scaled": auxiliary},
            "auxiliary_at": "self",
            "resample_method": method,
            "subsample": 7,
            "random_state": 42,
        }
        expected = points.cosample(raster, mask=raster_mask, **options)

        # Write the value raster and boolean raster mask to disk, then reopen both without loading their values
        raster_file, mask_file = tmp_path / "values.tif", tmp_path / "mask.tif"
        raster.to_file(raster_file)
        raster_mask.to_file(mask_file)
        unloaded = gu.Raster(raster_file, load_data=False)
        unloaded_mask = gu.Raster(mask_file, load_data=False, is_mask=True)

        # Read the unloaded raster and mask in Multiproc tiles while interpolating values at the selected points
        result = points.cosample(
            unloaded,
            mask=unloaded_mask,
            mp_config=MultiprocConfig(chunks=(5, 6), outfile=str(tmp_path / "unused-point-output.tif")),
            **options,
        )

        # 3/ Check that both rasters stay unloaded and the output preserves values, row order, and duplicate labels
        assert not unloaded.is_loaded
        assert not unloaded_mask.is_loaded
        assert len(result.ds) == 7
        assert_geodataframe_equal(result.ds, expected.ds)

    @pytest.mark.parametrize("subsample", [1, 8, 0.3])
    @pytest.mark.parametrize("mask_type", ["array", "raster", "vector"])
    def test_cosample__dask_point_partitions(self, subsample: int | float, mask_type: str) -> None:
        """Checks that values, labels and 3D geometry match eager across Dask point partitions."""

        import_optional("dask")
        import dask.array as da

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # 1/ Prepare point columns and independent geometry elevations on known raster pixels
        # Give geometry Z coordinates and data-column values different numbers so both can be checked independently
        _register_dask_pointcloud_accessor()
        positions = np.arange(40, dtype=float)
        raster = _raster(np.arange(120, dtype=float).reshape(10, 12))
        raster.data[3, 4] = np.ma.masked
        x, y = raster.ij2xy(1 + positions.astype(int) % 7, 1 + positions.astype(int) % 9)
        geometry = gpd.points_from_xy(x, y, z=positions + 500)
        first = gpd.GeoDataFrame({"height": positions.copy()}, geometry=geometry, crs=raster.crs)
        second = gpd.GeoDataFrame({"height": 2 * positions}, geometry=geometry, crs=raster.crs)
        first.index = np.tile(["a", "b", "a", "c"], 10)
        second.index = np.arange(40) + 100
        first.iloc[1, 0], second.iloc[2, 0] = np.nan, np.nan
        auxiliary = 3 * positions
        auxiliary[3] = np.nan
        mask: Any = positions.astype(int) % 5 != 0
        if mask_type == "raster":
            keep = np.indices(raster.shape)[1] % 4 != 0
            mask = gu.Raster.from_array(keep, raster.transform, raster.crs)
        elif mask_type == "vector":
            mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 8, 10)], crs=raster.crs))

        # 2/ Calculate an eager reference, then vary each Dask input's partitions or chunks
        options: dict[str, Any] = {
            "auxiliary_at": {"scaled": "self"},
            "resample_method": "nearest",
            "subsample": subsample,
            "random_state": 42,
        }
        eager_mask = mask
        if mask_type == "raster":
            eager_mask = gu.RasterAccessor.from_array(mask.data, mask.transform, mask.crs)
        expected = first.pc.cosample(
            second, auxiliary={"grid": raster.to_xarray(), "scaled": auxiliary}, mask=eager_mask, **options
        )
        for partitions in (2, 5):
            lazy_first = dgpd.from_geopandas(first, npartitions=partitions, sort=False)
            lazy_second = dgpd.from_geopandas(second, npartitions=partitions + 1, sort=False)
            lazy_raster = raster.to_xarray().chunk({"y": partitions + 1, "x": 4})
            lazy_auxiliary = da.from_array(auxiliary, chunks=7)
            # Give an array mask independent Dask chunks, or apply the spatial mask to each Dask point partition
            lazy_mask = mask
            if mask_type == "array":
                lazy_mask = da.from_array(mask, chunks=9)
            elif mask_type == "raster":
                lazy_mask = _raster(keep.astype(float)).to_xarray().astype(bool).chunk({"y": 3, "x": 5})
            result = lazy_first.pc.cosample(
                lazy_second,
                auxiliary={"grid": lazy_raster, "scaled": lazy_auxiliary},
                mask=lazy_mask,
                **options,
            )

            # 3/ Check that the output remains partitioned and its metadata is unchanged before comparing values
            assert isinstance(result, dgpd.GeoDataFrame)
            assert not result.pc.is_loaded
            assert result.pc.data_column == "self"
            assert not lazy_first.pc.is_loaded
            output = result.compute()
            assert list(output.columns) == ["self", "other", "grid", "scaled", "geometry"]
            assert np.array_equal(output.index.to_numpy(), expected.index.to_numpy())
            assert np.array_equal(output.geometry.to_numpy(), expected.geometry.to_numpy())
            assert np.array_equal(output.geometry.z.to_numpy(), expected.geometry.z.to_numpy())
            assert np.array_equal(output.drop(columns="geometry").to_numpy(), expected.drop(columns="geometry"))
            assert output.crs == expected.crs

    @pytest.mark.parametrize("subsample", [1, 8])
    def test_cosample__dask_points_defer_value_interpolation(
        self, subsample: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that Dask reads raster validity first and waits to interpolate the selected values."""

        import_optional("dask")
        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        import geoutils.interface.interpolation as interpolation
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # 1/ Prepare duplicate point labels and one raster nodata pixel to check validity and row order
        _register_dask_pointcloud_accessor()
        positions = np.arange(24, dtype=float)
        raster = _raster(np.arange(120, dtype=float).reshape(10, 12))
        raster.data[3, 4] = np.ma.masked
        x, y = raster.ij2xy(1 + positions.astype(int) % 7, 1 + positions.astype(int) % 9)
        points = gu.PointCloud.from_xyz(x, y, positions, crs=raster.crs).ds
        points.index = np.tile(["b", "a", "b"], 8)
        options: dict[str, Any] = {"resample_method": "nearest", "subsample": subsample, "random_state": 42}
        expected = points.pc.cosample(raster.to_xarray(), **options)
        lazy_points = dgpd.from_geopandas(points, npartitions=3, sort=False)
        lazy_raster = raster.to_xarray().chunk({"y": 4, "x": 5})

        # 2/ Check that building the Dask result reads raster validity without interpolating raster values
        calls: list[bool] = []
        original_interpolation = interpolation._interp_points_base

        def track_interpolation(*args: Any, **kwargs: Any) -> Any:
            """
            Record whether interpolation is checking raster validity or calculating raster values.
            """
            validity_only = kwargs.get("_validity_only", False)
            calls.append(validity_only)
            return original_interpolation(*args, **kwargs)

        monkeypatch.setattr(interpolation, "_interp_points_base", track_interpolation)
        result = lazy_points.pc.cosample(lazy_raster, **options)
        assert isinstance(result, dgpd.GeoDataFrame)
        assert not result.pc.is_loaded
        assert result.pc.data_column == "self"
        assert result.pc.bounds is None
        assert lazy_points.pc.data_column == points.pc.data_column
        assert calls and all(calls)

        # 3/ Compute values only on request, preserving the eager labels, coordinates and selected values
        output = result.compute()
        assert any(not validity_only for validity_only in calls)
        assert np.array_equal(output.index.to_numpy(), expected.index.to_numpy())
        assert np.array_equal(output.geometry.to_numpy(), expected.geometry.to_numpy())
        assert np.array_equal(output.drop(columns="geometry").to_numpy(), expected.drop(columns="geometry"))

        # Asking for a row count may compute the result, but must describe selected rows rather than the source
        assert result.pc.point_count == len(expected)
        assert lazy_points.pc.point_count == len(points)

    @pytest.mark.parametrize("lazy_input", ["raster", "points"])
    @pytest.mark.parametrize("caller", ["raster", "points"])
    def test_cosample__mixed_eager_and_dask_inputs(self, lazy_input: str, caller: str) -> None:
        """Checks that eager and Dask raster and point inputs can be used together."""

        import_optional("dask")
        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # Place observations on known raster pixels so interpolation and point values have exact references
        _register_dask_pointcloud_accessor()
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        rows, columns = np.array([1, 2, 3]), np.array([1, 2, 3])
        x, y = raster.ij2xy(rows, columns)
        values = np.array([10.0, 20.0, 30.0])
        points = gu.PointCloud.from_xyz(x, y, values, crs=raster.crs).ds

        # Use accessor-compatible raster and point inputs while making only one of them Dask-backed
        raster_input = raster.to_xarray()
        point_input = points
        if lazy_input == "raster":
            raster_input = raster_input.chunk({"y": 2, "x": 3})
        else:
            point_input = dgpd.from_geopandas(points, npartitions=2, sort=False)
        source, other = (raster_input.rst, point_input) if caller == "raster" else (point_input.pc, raster_input)
        result = source.cosample(other, resample_method="nearest")

        # Calculate the same caller order from eager accessor inputs
        eager_raster = raster.to_xarray()
        eager_source, eager_other = (eager_raster.rst, points) if caller == "raster" else (points.pc, eager_raster)
        expected = eager_source.cosample(eager_other, resample_method="nearest")

        # Check input and output loading before comparing every row with eager
        if lazy_input == "raster":
            da = pytest.importorskip("dask.array")
            assert isinstance(raster_input.data, da.Array)
        else:
            assert not point_input.pc.is_loaded and not result.pc.is_loaded
        output = result.compute() if lazy_input == "points" else result
        assert_geodataframe_equal(output, expected)
        raster_column, point_column = ("self", "other") if caller == "raster" else ("other", "self")
        assert np.array_equal(output[raster_column], raster.data[rows, columns])
        assert np.array_equal(output[point_column], values)
        assert output.geometry.equals(points.geometry)


class TestCosampleErrors:
    """
    Test module for errors raised by cosample() when inputs cannot produce a clear result.

    We test the following cases:
    - Invalid band indexes or columns names raise an error, without loading objects,
    - Geospatial inputs with incompatible types (Xarray/Pandas vs GeoUtils objects) and empty common support
        (no intersection, different CRS with reprojection option turned off) both raise an error.
    - Consistency of common support definition through `at` and `raster_point_mode`.
    """

    def test_cosample__error_basic_inputs(self) -> None:
        """Checks that wrong inputs raise clear errors."""

        # Create two grids and one point set so a conversion direction alone cannot choose one grid
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        other = _raster(np.ones((4, 5)), x_origin=1)
        points = raster.to_pointcloud()

        # Check that raster_point_mode inputs that are ambiguous or conflict with "at" raise proper errors
        with pytest.raises(ValueError, match="unambiguous"):
            raster.cosample(other, raster_point_mode="grid_points")
        with pytest.raises(ValueError, match="conflicts"):
            raster.cosample(points, at=points, raster_point_mode="grid_points")
        with pytest.raises(ValueError, match="Argument ``raster_point_mode`` must"):
            raster.cosample(points, raster_point_mode="unknown")

        # Check error that reduce points is not yet implemented
        with pytest.raises(NotImplementedError, match="revision of Raster.reduce_points"):
            raster.cosample(points, raster_point_mode="resample_raster", resample_method="reduce")

        # Check errors for passing similarly named arguments of grid/resample kwargs and cosample()
        # (some must be passed directly to cosample(), not kwargs)
        with pytest.raises(ValueError, match="outside ``grid_kwargs``"):
            raster.cosample(points, grid_kwargs={"resampling": "nearest"})
        with pytest.raises(ValueError, match="outside ``resample_kwargs``"):
            raster.cosample(points, resample_kwargs={"points": (np.ones(1), np.ones(1))})

    @pytest.mark.parametrize("argument", ["self", "other", "auxiliary"])
    @pytest.mark.parametrize("output_support", ["raster", "points"])
    @pytest.mark.parametrize("invalid_band,error", [("height", TypeError), (0, ValueError), (3, ValueError)])
    def test_cosample__error_invalid_bands_do_not_load_rasters(
        self, argument: str, output_support: str, invalid_band: Any, error: type[Exception], tmp_path: Path
    ) -> None:
        """Checks that invalid primary or auxiliary bands raise error from raster metadata before loading."""

        # Write two band raster, so we can request band 0 and 3 and check for errors below
        values = np.arange(20, dtype=float).reshape(4, 5)
        raster = _raster(np.stack((values, values + 100)))
        path = tmp_path / "two_bands.tif"
        raster.to_file(path)
        x, y = raster.ij2xy(np.array([1, 2]), np.array([1, 2]))
        points = gu.PointCloud.from_xyz(x, y, np.ones(2), crs=raster.crs)
        source = gu.Raster(path, load_data=False)

        # Put the invalid selection in one argument while all other bands and output locations remain valid
        options: dict[str, Any] = {"at": points if output_support == "points" else "self"}
        if argument == "auxiliary":
            options["auxiliary"] = {"extra": (source, invalid_band)}
        else:
            options["band" if argument == "self" else "other_band"] = invalid_band
        with pytest.raises(error, match="band|selector"):
            source.cosample(source, **options)

        # The error must be raised before reading the input
        assert not source.is_loaded

    @pytest.mark.parametrize(
        "auxiliary_at", [{"missing": "self"}, {"extra": "unknown"}, {"extra": None}, {"extra": 1}, "unknown"]
    )
    def test_cosample__error_invalid_auxiliary_locations(self, auxiliary_at: Any) -> None:
        """Checks that unknown auxiliary names and invalid location choices raise errors."""

        # Use an auxiliary for which the coordinates make auxiliary_at unnecessary
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        auxiliary = {"extra": (raster, 1)}

        # Validate every supplied location choice so a typo cannot silently survive input preparation
        with pytest.raises(ValueError, match="auxiliary_at"):
            raster.cosample(raster, auxiliary=auxiliary, auxiliary_at=auxiliary_at)

    @pytest.mark.parametrize("output_support", ["raster", "points"])
    @pytest.mark.parametrize(
        "native_support,shape",
        [
            ("raster", (12,)),
            ("raster", (4, 3)),
            ("raster", (2, 3, 4)),
            ("points", (12, 1)),
            ("points", (1, 12)),
            ("points", (11,)),
        ],
    )
    def test_cosample__error_raw_auxiliary_shape(
        self, output_support: str, native_support: str, shape: tuple[int, ...]
    ) -> None:
        """Checks that raw arrays match their native grid or point shape before any conversion to output locations."""

        # Use twelve grid pixels and twelve corresponding points (so that equal counts cannot hide the wrong shape)
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        points = raster.to_pointcloud()
        source = raster if native_support == "raster" else points
        support = raster if output_support == "raster" else points
        auxiliary = np.ones(shape)

        # Raise error for flattened grids, extra point dimensions, multiple raw bands and arrays with the wrong size
        with pytest.raises(ValueError, match="(?i)shape|native|point"):
            source.cosample(source, auxiliary={"extra": auxiliary}, auxiliary_at="self", at=support)

    @pytest.mark.parametrize("caller", ["raster", "points"])
    @pytest.mark.parametrize("accessor", [False, True])
    @pytest.mark.parametrize("input_type", ["raster", "points"])
    @pytest.mark.parametrize("argument", ["other", "auxiliary", "selected_auxiliary", "at", "mask"])
    def test_cosample__error_mixed_spatial_container_families(
        self, caller: str, accessor: bool, input_type: str, argument: str
    ) -> None:
        """Checks that inputs all use the same "family" (Xarray/Pandas or GeoUtils objects)."""

        # Give the caller valid point locations that can also be used by each input
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        points = raster.to_pointcloud()
        source = raster if caller == "raster" else points
        if accessor:
            source = raster.to_xarray().rst if caller == "raster" else points.ds.pc
        other = points.ds if accessor else points
        incompatible: Any = raster if input_type == "raster" else points

        # Use boolean values for masks so only their container family is invalid
        if argument == "mask":
            if input_type == "raster":
                incompatible = gu.Raster.from_array(np.ones(raster.shape, dtype=bool), raster.transform, raster.crs)
            else:
                incompatible = points.copy(new_array=np.ones(points.point_count, dtype=bool))
        if not accessor:
            if input_type == "raster":
                incompatible = gu.RasterAccessor.from_array(incompatible.data, incompatible.transform, incompatible.crs)
            else:
                incompatible = incompatible.ds

        # Put the incompatible input in one argument while all other inputs use the caller's container type
        options: dict[str, Any] = {}
        if argument == "other":
            other = incompatible
        elif argument == "auxiliary":
            options["auxiliary"] = {"extra": incompatible}
        elif argument == "selected_auxiliary":
            selector = 1 if input_type == "raster" else None
            options["auxiliary"] = {"extra": (incompatible, selector)}
        else:
            options[argument] = incompatible
        with pytest.raises(TypeError, match="mix"):
            source.cosample(other, **options)

    @pytest.mark.parametrize("mismatch", ["shifted", "reordered", "shortened"])
    def test_cosample__error_point_locations_before_raster_preparation(
        self, mismatch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that incompatible auxiliary points raise before any raster alignment or interpolation occurs."""

        # Use point support in a different CRS so processing the raster would require reprojection
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        points = raster.to_pointcloud().reproject(crs=4326)
        x, y = points.ds.geometry.x.to_numpy(), points.ds.geometry.y.to_numpy()
        values = np.arange(points.point_count, dtype=float)
        if mismatch == "shifted":
            x = x + 0.1
        elif mismatch == "reordered":
            x, y, values = x[::-1], y[::-1], values[::-1]
        else:
            x, y, values = x[:-1], y[:-1], values[:-1]
        auxiliary = gu.PointCloud.from_xyz(x, y, values, crs=points.crs)

        # Fail if expensive spatial preparation starts before all point locations have been checked
        def unexpected_raster_operation(*args: Any, **kwargs: Any) -> Any:
            """Report raster work that must not occur for incompatible point inputs."""

            pytest.fail("Point coordinates must be checked before raster preparation.")

        monkeypatch.setattr(gu.Raster, "reproject", unexpected_raster_operation)
        monkeypatch.setattr(gu.Raster, "interp_points", unexpected_raster_operation)
        with pytest.raises(ValueError, match="ordered support coordinates"):
            raster.cosample(points, auxiliary={"extra": auxiliary}, align="reproject")

    @pytest.mark.parametrize("lazy_input", ["self", "other", "auxiliary", "mask", "at"])
    def test_cosample__error_multiproc_with_dask_inputs(self, lazy_input: str, tmp_path: Path) -> None:
        """Checks that Dask inputs cannot be combined with Multiproc output."""

        # Pass a Dask input in each input position while the other arguments are eager
        import_optional("dask")
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        eager = raster.to_xarray()
        lazy = eager.chunk({"y": 2, "x": 3})
        source = lazy.rst if lazy_input == "self" else eager.rst
        other = lazy if lazy_input == "other" else eager
        options: dict[str, Any] = {}
        if lazy_input == "auxiliary":
            options["auxiliary"] = {"extra": lazy}
        elif lazy_input == "mask":
            options["mask"] = lazy > 0
        elif lazy_input == "at":
            options["at"] = lazy

        # Raise an error without writing a Multiproc result (ensures files are properly cleaned at any breakpoints)
        outfile = tmp_path / "should-not-exist.tif"
        with pytest.raises(ValueError, match="Multiprocessing and Dask"):
            source.cosample(other, mp_config=MultiprocConfig(chunks=(2, 3), outfile=str(outfile)), **options)
        assert not outfile.exists()

    @pytest.mark.parametrize(
        "option,argument",
        [
            ("grid_kwargs", "mp_config"),
            ("grid_kwargs", "data_column"),
            ("resample_kwargs", "mp_config"),
            ("resample_kwargs", "_validity_only"),
        ],
    )
    def test_cosample__error_reserved_backend_and_validity_options(self, option: str, argument: str) -> None:
        """Checks that internal options cannot be passed through interpolation keywords."""

        # Use only valid raster values so the misplaced option is the sole cause of failure
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        options: dict[str, Any] = {option: {argument: None}}
        with pytest.raises(ValueError, match=f"outside ``{option}``"):
            raster.cosample(raster, **options)

    @pytest.mark.parametrize("name", ["self", "other", "geometry"])
    def test_cosample__error_reserved_auxiliary_names(self, name: str) -> None:
        """Checks that auxiliary names cannot replace a primary value or the point geometry column."""

        # Use an otherwise valid call so only the added column name is invalid
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        with pytest.raises(ValueError, match="cannot be"):
            raster.cosample(raster, auxiliary={name: raster})

    def test_cosample__error_conflicting_raster_point_mode(self) -> None:
        """Checks that an explicit point resampling mode rejects a raster output target."""

        # Request point output while explicitly selecting a raster grid
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        points = raster.to_pointcloud()
        with pytest.raises(ValueError, match="conflicts"):
            raster.cosample(points, at="self", raster_point_mode="resample_raster")

    def test_cosample__error_empty_common_support(self) -> None:
        """Checks that an empty common sample raises before creating an unusable spatial output."""

        # Remove every available raster pixel or point through the user mask
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        points = raster.to_pointcloud()
        with pytest.raises(ValueError, match="no finite data common"):
            raster.cosample(raster, mask=np.zeros(raster.shape, dtype=bool))
        with pytest.raises(ValueError, match="no finite data common"):
            points.cosample(points, mask=np.zeros(len(points.ds), dtype=bool))

    def test_cosample__error_grid_crs_alignment(self) -> None:
        """Checks that gridding rejects points in another CRS while alignment is disabled."""

        # Reproject points away from the raster CRS
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        points = raster.to_pointcloud().reproject(crs=4326)

        # Require an explicit request before moving the points back onto the raster grid
        with pytest.raises(ValueError, match="support CRS"):
            raster.cosample(points, raster_point_mode="grid_points", grid_method="nearest")

    @pytest.mark.parametrize("at", ["self", "other", "explicit"])
    def test_cosample__error_raster_grid_alignment(self, at: str) -> None:
        """Checks that mismatched raster grids require an explicit request for alignment."""

        # Shift the second raster by one pixel and select each available support form
        first = _raster(np.arange(12, dtype=float).reshape(3, 4))
        second = _raster(np.arange(12, dtype=float).reshape(3, 4), x_origin=1)
        selected_at = second if at == "explicit" else at

        # Reject the mismatched grids while alignment is disabled
        with pytest.raises(ValueError, match="does not share"):
            first.cosample(second, at=selected_at)
