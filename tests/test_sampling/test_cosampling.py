"""Tests for cosampling on raster grids and point geometries through the public spatial API."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._typing import NDArrayNum


def _raster(data: NDArrayNum, *, x_origin: float = 0) -> gu.Raster:
    """Create a unit grid so row and column positions give simple expected sample values."""

    return gu.Raster.from_array(data, transform=from_origin(x_origin, data.shape[-2], 1, 1), crs=32633, nodata=-99999)


class TestRasterPointModes:
    """Conversion direction, method options and target selection for mixed inputs."""

    @pytest.mark.parametrize("caller", ["raster", "points"])
    @pytest.mark.parametrize("accessor", [False, True])
    @pytest.mark.parametrize("explicit_at", [False, True])
    def test_grid_points(self, caller: str, accessor: bool, explicit_at: bool) -> None:
        """Checks that circular means populate the selected grid with a common mask in either caller order."""

        # Place two observations around each grid location with a known mean and no neighboring pixels in range
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

        # Infer the raster target from the mode, or infer the mode from an explicit target
        first, second = (raster, points) if caller == "raster" else (points, raster)
        source = first
        if accessor:
            source = first.to_xarray().rst if caller == "raster" else first.ds.pc
        target = {"at": raster} if explicit_at else {"raster_point_mode": "grid_points"}
        result = source.cosample(
            second,
            **target,
            grid_method="mean",
            grid_kwargs={"dist_nodata_pixel": 0.4, "min_points": 2},
            auxiliary={"offset": point_values + 10},
            auxiliary_at="other" if caller == "raster" else "self",
            mask=keep,
        )

        # Check numerical means, raw point auxiliaries and the shared mask without relying on the grid implementation
        assert isinstance(result, xr.DataArray if accessor else gu.Raster)
        output = result.rst if accessor else result
        data = result.values if accessor else result.data.filled(np.nan)
        expected_bands = [expected + 100, expected] if caller == "raster" else [expected, expected + 100]
        expected_bands.append(expected + 10)
        assert output.transform == raster.transform
        np.testing.assert_allclose(data, np.where(keep, np.stack(expected_bands), np.nan), atol=1e-12)
        np.testing.assert_array_equal(points.data, point_values)

    @pytest.mark.parametrize("caller", ["raster", "points"])
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_resample_raster(self, caller: str, method: str) -> None:
        """Checks that raster resampling evaluates an affine surface at irregular target coordinates."""

        # An affine surface has exact bilinear values away from the raster edge
        rows, columns = np.indices((7, 9))
        raster = _raster((10 * rows + 2 * columns).astype(float))
        target_rows = np.array([1.2, 2.3, 4.1])
        target_columns = np.array([1.1, 3.2, 5.4])
        x, y = raster.ij2xy(target_rows, target_columns)
        points = gu.PointCloud.from_xyz(x, y, np.array([5.0, 6.0, 7.0]), crs=raster.crs)

        # Choose conversion direction independently of which spatial type owns the method call
        first, second = (raster, points) if caller == "raster" else (points, raster)
        result = first.cosample(second, raster_point_mode="resample_raster", resample_method=method)

        # Linear resampling reproduces the plane, while nearest uses the closest grid coordinates
        expected = 10 * target_rows + 2 * target_columns
        if method == "nearest":
            expected = 10 * np.rint(target_rows) + 2 * np.rint(target_columns)
        raster_column = "self" if caller == "raster" else "other"
        np.testing.assert_allclose(result.ds[raster_column], expected)
        assert result.ds.geometry.equals(points.ds.geometry)

    def test_grid_crs_alignment(self) -> None:
        """Checks that gridding permits coordinate conversion only when alignment is explicitly enabled."""

        # Preserve known pixel observations while representing their coordinates in a different CRS
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        points = raster.to_pointcloud().reproject(crs=4326)

        # Require the same explicit CRS alignment policy as the raster resampling workflow
        with pytest.raises(ValueError, match="support CRS"):
            raster.cosample(points, raster_point_mode="grid_points", grid_method="nearest")
        result = raster.cosample(points, raster_point_mode="grid_points", grid_method="nearest", align="reproject")

        # Returning to the original grid recovers every observation within floating point coordinate precision
        np.testing.assert_allclose(result.data[0], result.data[1])

    @pytest.mark.parametrize("chunks", [(7, 11), (32, 47), (256, 256)])
    @pytest.mark.parametrize("caller", ["raster", "points"])
    def test_dask_gridding_chunks(self, chunks: tuple[int, int], caller: str, tmp_path: Path) -> None:
        """Checks that lazy point gridding preserves values and seeded sample locations across chunk sizes."""

        # Each point coincides with a known grid coordinate so nearest gridding has an exact array reference
        pytest.importorskip("dask.array")
        pytest.importorskip("dask_geopandas")
        values = np.arange(65 * 97, dtype=float).reshape(65, 97)
        raster = _raster(values)
        points = raster.to_pointcloud()
        lazy_raster = raster.to_xarray().chunk({"y": chunks[0], "x": chunks[1]})
        point_file = tmp_path / "observations.gpkg"
        points.to_file(point_file)
        lazy_points = gu.open_pointcloud(str(point_file), data_column=points.data_column, chunks=1400)

        # Use actual lazy point partitions and select the same bounded subset through either accessor
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

        # The result remains lazy, and both its selected cells and finite values match eager grid sampling
        assert result.data.chunks is not None
        np.testing.assert_allclose(result.compute().values, expected.data.filled(np.nan))

    def test_resampling_nodata_options(self) -> None:
        """Checks that resampling nodata options govern both point eligibility and the resulting values."""

        # One invalid neighbor distinguishes strict propagation from using the finite interpolation weights
        values = np.ones((5, 6), dtype=float)
        values[2, 2] = np.nan
        raster = _raster(values)
        x, y = raster.ij2xy(np.array([1.25, 3.0]), np.array([1.25, 3.0]))
        points = gu.PointCloud.from_xyz(x, y, np.array([10.0, 20.0]), crs=raster.crs)

        # Evaluate the same coordinates with two explicitly different nodata policies
        ignored = raster.cosample(points, resample_kwargs={"nodata_propagation": "ignore"})
        propagated = raster.cosample(points, resample_kwargs={"nodata_propagation": "propagate"})

        # Finite neighbors all equal one; strict propagation excludes only the point touching the gap
        assert list(ignored.ds.index) == [0, 1]
        assert list(propagated.ds.index) == [1]
        np.testing.assert_allclose(ignored.ds["self"], 1)
        np.testing.assert_allclose(propagated.ds["self"], 1)

    def test_mode_validation_and_reduction_placeholder(self) -> None:
        """Checks that ambiguous targets, contradictory modes and deferred window reduction raise clear errors."""

        # Use two raster grids and one point set to distinguish direction from exact target selection
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        other = _raster(np.ones((4, 5)), x_origin=1)
        points = raster.to_pointcloud()

        # A conversion mode cannot identify a unique target among several grids or override explicit points
        with pytest.raises(ValueError, match="unambiguous"):
            raster.cosample(other, raster_point_mode="grid_points")
        with pytest.raises(ValueError, match="conflicts"):
            raster.cosample(points, at=points, raster_point_mode="grid_points")
        with pytest.raises(ValueError, match="raster_point_mode must"):
            raster.cosample(points, raster_point_mode="unknown")

        # Keep reduction discoverable without implying that the existing reduce_points API has changed
        with pytest.raises(NotImplementedError, match="revision of Raster.reduce_points"):
            raster.cosample(points, raster_point_mode="resample_raster", resample_method="reduce")
        with pytest.raises(ValueError, match="outside grid_kwargs"):
            raster.cosample(points, grid_kwargs={"resampling": "nearest"})
        with pytest.raises(ValueError, match="outside resample_kwargs"):
            raster.cosample(points, resample_kwargs={"points": (np.ones(1), np.ones(1))})


class TestRasterSupport:
    """Common masks, band selection and output grids for raster comparisons."""

    @pytest.mark.parametrize("accessor", [False, True])
    def test_joint_validity_and_auxiliary(self, accessor: bool) -> None:
        """Checks that every output band shares the finite primary, auxiliary and user mask intersection."""

        # Give each input a different excluded cell to distinguish their contributions to the common mask
        first = np.arange(20, dtype=float).reshape(4, 5)
        second, auxiliary = 10 * first, 100 * first
        first[0, 0], second[1, 1], auxiliary[2, 2] = np.nan, np.nan, np.nan
        mask = np.ones(first.shape, dtype=bool)
        mask[3, 3] = False
        raster = _raster(first)

        # Exercise the same public operation through both interfaces
        source = raster.to_xarray().rst if accessor else raster
        result = source.cosample(_raster(second), auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=mask)
        assert isinstance(result, xr.DataArray if accessor else gu.Raster)
        output = result.rst if accessor else result
        data = result.to_numpy() if accessor else result.data.filled(np.nan)

        # Preserve the chosen grid and describe the primary and auxiliary band order
        assert output.shape == raster.shape
        assert output.transform == raster.transform
        assert output.crs == raster.crs
        assert output.tags["long_name"] == ("self", "other", "aux")
        expected = mask & np.isfinite(first) & np.isfinite(second) & np.isfinite(auxiliary)
        np.testing.assert_array_equal(np.isfinite(data), np.broadcast_to(expected, data.shape))

        # Compare all retained values against the original inputs, including zero
        np.testing.assert_array_equal(data[0, expected], first[expected])
        np.testing.assert_array_equal(data[1, expected], second[expected])
        np.testing.assert_array_equal(data[2, expected], auxiliary[expected])

    @pytest.mark.parametrize("at", ["self", "other", "explicit"])
    def test_output_grid_and_alignment(self, at: str) -> None:
        """Checks that at selects the output grid and mismatched inputs require explicit reprojection."""

        # Shift the second raster by one cell so the grids overlap without being identical
        first = _raster(np.arange(12, dtype=float).reshape(3, 4))
        second = _raster(np.arange(12, dtype=float).reshape(3, 4), x_origin=1)
        support = first if at == "self" else second
        selected_at = second if at == "explicit" else at

        # Require the user to opt into alignment on the selected support
        with pytest.raises(ValueError, match="does not share"):
            first.cosample(second, at=selected_at)
        result = first.cosample(second, at=selected_at, align="reproject")

        # Keep the chosen transform and mask cells outside the common extent
        assert isinstance(result, gu.Raster)
        assert result.transform == support.transform
        assert result.shape == support.shape
        assert result.count == 2
        assert np.count_nonzero(~np.ma.getmaskarray(result.data[0])) == 9

    def test_selected_bands_and_auxiliary_owners(self) -> None:
        """Checks that requested bands and auxiliaries tied to either input retain their documented order."""

        # Offset the second band so accidentally reading the first band changes the result
        base = np.arange(20, dtype=float).reshape(4, 5)
        first = _raster(np.stack((base, base + 100)))
        second = _raster(np.stack((2 * base, 2 * base + 200)))

        # Bind each plain auxiliary array to its own primary dataset
        result = first.cosample(
            second,
            band=2,
            other_band=2,
            auxiliary={"first_aux": base + 1, "second_aux": 2 * base + 2},
            auxiliary_at={"first_aux": "self", "second_aux": "other"},
        )

        # Read the combined raster through its usual band API
        bands = result.split_bands()
        expected = (base + 100, 2 * base + 200, base + 1, 2 * base + 2)
        for band, values in zip(bands, expected):
            np.testing.assert_array_equal(band.data, values)
        assert result.tags["long_name"] == ("self", "other", "first_aux", "second_aux")

    def test_geodataframe_mask(self) -> None:
        """Checks that a GeoDataFrame mask keeps only grid cells within its geometries."""

        # Cover the first two columns with a rectangle in the raster CRS
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        geometry = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 4)], crs=raster.crs)
        result = raster.cosample(raster + 1, mask=geometry)

        # Verify spatial membership from the grid rather than a custom sample index
        expected = np.zeros(raster.shape, dtype=bool)
        expected[:, :2] = True
        np.testing.assert_array_equal(~np.ma.getmaskarray(result.data[0]), expected)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("raster_mask", [False, True])
    def test_masked_integers_and_boolean_masks(self, lazy: bool, raster_mask: bool) -> None:
        """Checks that masked primaries, auxiliaries and masks exclude the same output cells."""

        # Give each input a different missing cell so no mask can be silently discarded
        data = np.ma.array(np.arange(30, dtype=np.int32).reshape(5, 6), mask=False)
        data.mask[0, 0] = True
        auxiliary = np.ma.array(2 * data.data, mask=False)
        auxiliary.mask[1, 1] = True
        mask = np.ma.array(np.ones(data.shape, dtype=bool), mask=False)
        mask.mask[2, 2], mask[3, 3] = True, False
        first = _raster(data)
        selected_mask = first.from_array(mask, first.transform, first.crs) if raster_mask else mask

        # Exercise lazy alignment with the same eager masked auxiliary and mask
        source = first
        if lazy:
            da = pytest.importorskip("dask.array")
            source = gu.RasterAccessor.from_array(
                da.from_array(data.astype(float).filled(np.nan), chunks=(2, 3)), first.transform, first.crs
            ).rst
        result = source.cosample(source, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=selected_mask)
        output = result.to_numpy() if lazy else result.data.filled(np.nan)

        # Compare the common mask and retained values independently of the sampler
        expected = ~data.mask & ~auxiliary.mask & mask.filled(False)
        np.testing.assert_array_equal(np.isfinite(output[0]), expected)
        np.testing.assert_array_equal(output[0, expected], data.data[expected])
        np.testing.assert_array_equal(output[2, expected], auxiliary.data[expected])

    @pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
    @pytest.mark.parametrize("accessor", [False, True])
    def test_singleton_spatial_dimensions(self, shape: tuple[int, int], accessor: bool) -> None:
        """Checks that one row or column remains a spatial dimension in a combined multiband raster."""

        # Use a sparse mask even when the support has only one row or column
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
        raster = _raster(values)
        source = gu.RasterAccessor.from_array(values, raster.transform, raster.crs).rst if accessor else raster
        mask = values % 2 == 0
        result = source.cosample(source, mask=mask)

        # Retain both spatial axes separately from the two output bands
        data = result.to_numpy() if accessor else result.data.filled(np.nan)
        assert data.shape == (2, *shape)
        np.testing.assert_array_equal(np.isfinite(data[0]), mask)
        np.testing.assert_array_equal(data[0, mask], values[mask])

    @pytest.mark.parametrize("subsample", [1, 12, 0.25])
    def test_dask_output_and_chunk_independence(self, subsample: int | float) -> None:
        """Checks that raster outputs stay lazy and topk selects the same cells across chunk layouts."""

        # Use distinct chunk layouts for the two primary rasters
        da = pytest.importorskip("dask.array")
        array = np.arange(63, dtype=float).reshape(7, 9)
        transform = from_origin(0, 7, 1, 1)
        first = gu.RasterAccessor.from_array(da.from_array(array, chunks=(2, 4)), transform, 32633)
        second = gu.RasterAccessor.from_array(da.from_array(2 * array, chunks=(4, 3)), transform, 32633)

        # Repeat the sample after changing the first input's chunks
        result = first.rst.cosample(second, subsample=subsample, random_state=42, strategy="topk")
        changed = first.chunk({"y": 4, "x": 3}).rst.cosample(
            second, subsample=subsample, random_state=42, strategy="topk"
        )
        assert isinstance(result, xr.DataArray)
        assert isinstance(result.data, da.Array)
        assert isinstance(first.data, da.Array)

        # Compare eager and lazy selection as well as the values in each retained cell
        eager = _raster(array).cosample(_raster(2 * array), subsample=subsample, random_state=42, strategy="topk")
        output = result.compute().to_numpy()
        np.testing.assert_allclose(output, changed.to_numpy(), equal_nan=True)
        np.testing.assert_allclose(output, eager.data.filled(np.nan), equal_nan=True)
        count = array.size if subsample == 1 else int(subsample * array.size) if subsample < 1 else subsample
        assert np.count_nonzero(np.isfinite(output[0])) == count
        np.testing.assert_allclose(output[1], 2 * output[0], equal_nan=True)


class TestPointSupport:
    """Selected point geometries, named columns and raster interpolation."""

    @pytest.mark.parametrize("caller", ["raster", "pointcloud"])
    @pytest.mark.parametrize("accessor", [False, True])
    def test_mixed_inputs(self, caller: str, accessor: bool) -> None:
        """Checks that a mixed comparison returns point support through either calling interface."""

        # Exclude one point through each primary's finite data mask
        values = np.arange(30, dtype=float).reshape(5, 6)
        values[1, 2] = np.nan
        raster = _raster(values)
        rows, columns = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
        x, y = raster.ij2xy(rows, columns)
        points = gu.PointCloud.from_xyz(x, y, np.array([4.0, 5.0, 6.0, np.nan]), crs=raster.crs)
        points.ds.index = ["a", "b", "c", "d"]

        # Choose the caller independently of the support and output representation
        source, other = (raster, points) if caller == "raster" else (points, raster)
        if accessor:
            source = source.to_xarray().rst if caller == "raster" else source.ds.pc
        result = source.cosample(other, resample_method="nearest")
        assert isinstance(result, gpd.GeoDataFrame if accessor else gu.PointCloud)
        output = result if accessor else result.ds

        # Preserve original point labels and geometry alongside named primary columns
        np.testing.assert_array_equal(output.index, ["a", "c"])
        assert output.geometry.equals(points.ds.geometry.iloc[[0, 2]])
        assert output.pc.data_column == "self"
        raster_values = values[rows[[0, 2]], columns[[0, 2]]]
        expected = (raster_values, [4.0, 6.0]) if caller == "raster" else ([4.0, 6.0], raster_values)
        np.testing.assert_array_equal(output["self"], expected[0])
        np.testing.assert_array_equal(output["other"], expected[1])

    @pytest.mark.parametrize("at", [None, "self", "other", "explicit"])
    def test_point_auxiliaries_and_support_labels(self, at: str | None) -> None:
        """Checks that point cosampling preserves the chosen support's labels and aligned auxiliary columns."""

        # Use duplicate labels and three dimensional geometry to detect lost support information
        positions = np.arange(8, dtype=float)
        first = gu.PointCloud.from_xyz(positions, positions**2, positions, crs=32633, use_z=True)
        second = gu.PointCloud.from_xyz(positions, positions**2, 2 * positions, crs=32633, use_z=True)
        first.ds.index = ["a", "b", "a", "c", "d", "e", "f", "g"]
        second.ds.index = np.arange(8) + 10
        auxiliary = 3 * positions
        auxiliary[3] = np.nan

        # Select the same ordered coordinates with a distinct index on the other input
        selected_at = second if at == "explicit" else at
        result = first.cosample(second, auxiliary={"weight": auxiliary}, auxiliary_at="self", at=selected_at)
        support = second if at in {"other", "explicit"} else first
        expected = np.array([0, 1, 2, 4, 5, 6, 7])

        # Keep the selected support geometry, including Z, and the fixed column order
        assert result.ds.geometry.equals(support.ds.geometry.iloc[expected])
        assert list(result.ds.columns) == ["self", "other", "weight", "geometry"]
        np.testing.assert_array_equal(result.ds["other"], 2 * result.ds["self"])
        np.testing.assert_array_equal(result.ds["weight"], 3 * result.ds["self"])

    def test_explicit_point_support_for_rasters(self) -> None:
        """Checks that an explicit point support yields compact columns for two raster inputs."""

        # Select a few known grid locations without using either raster as the output support
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        x, y = raster.ij2xy(np.array([1, 2, 3]), np.array([1, 2, 3]))
        support = gu.PointCloud.from_xyz(x, y, np.zeros(3), crs=raster.crs)
        result = raster.cosample(raster + 5, at=support, resample_method="nearest")

        # Check both values at the requested coordinates using ordinary point cloud columns
        assert isinstance(result, gu.PointCloud)
        np.testing.assert_array_equal(result.ds["self"], [7, 14, 21])
        np.testing.assert_array_equal(result.ds["other"], [12, 19, 26])
        assert result.ds.geometry.equals(support.ds.geometry)

    @pytest.mark.parametrize("mask_mode", ["inside", "outside"])
    def test_vector_mask(self, mask_mode: str) -> None:
        """Checks that a vector mask selects point geometries inside or outside its extent."""

        # Place two points inside a rectangle and two outside it
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        rows, columns = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
        x, y = raster.ij2xy(rows, columns)
        points = gu.PointCloud.from_xyz(x, y, np.arange(4, dtype=float), crs=raster.crs)
        mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 3, 3, 6)], crs=raster.crs))
        result = points.cosample(raster, mask=mask, mask_mode=mask_mode, resample_method="nearest")

        # Compare the retained point labels and both primary values
        expected = np.array([0, 1]) if mask_mode == "inside" else np.array([2, 3])
        np.testing.assert_array_equal(result.ds.index, expected)
        np.testing.assert_array_equal(result.ds["self"], expected)
        np.testing.assert_array_equal(result.ds["other"], raster.data[rows[expected], columns[expected]])

    def test_masked_auxiliary_and_mask(self) -> None:
        """Checks that raw point arrays retain missing values before common support is sampled."""

        # Exclude distinct points through an auxiliary mask, a mask's missing value and a false value
        positions = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, positions, crs=32633)
        auxiliary = np.ma.array(np.arange(5), mask=[False, True, False, False, False])
        mask = np.ma.array([True, True, True, False, True], mask=[False, False, True, False, False])
        result = points.cosample(points, auxiliary={"aux": auxiliary}, auxiliary_at="self", mask=mask)

        # Retain only the first and last points in their original order
        np.testing.assert_array_equal(result.ds.index, [0, 4])
        np.testing.assert_array_equal(result.ds["aux"], [0, 4])

    @pytest.mark.parametrize("accessor", [False, True])
    def test_selected_band_validity(self, accessor: bool) -> None:
        """Checks that point sampling uses validity from only the requested raster band."""

        # Place gaps at different points in each band to detect a mistaken validity band
        data = np.arange(30, dtype=float).reshape(5, 6)
        bands = np.stack((data, data + 100))
        bands[0, 1, 1], bands[1, 3, 3] = np.nan, np.nan
        raster = _raster(bands)
        x, y = raster.ij2xy(np.array([1, 2, 3]), np.array([1, 2, 3]))
        points = gu.PointCloud.from_xyz(x, y, np.array([1.0, 2.0, 3.0]), crs=raster.crs)

        # Request the second band through both raster interfaces
        source = raster.to_xarray().rst if accessor else raster
        result = source.cosample(points, band=2, resample_method="nearest")
        output = result if accessor else result.ds

        # Retain the point missing only from the first band
        np.testing.assert_array_equal(output.index, [0, 1])
        np.testing.assert_array_equal(output["self"], [107, 114])
        np.testing.assert_array_equal(output["other"], [1, 2])

    def test_subsampling_and_independent_storage(self) -> None:
        """Checks that subsampling preserves point order and returned values can change independently of inputs."""

        # Keep duplicate labels so sampling must follow positions rather than sorting index labels
        positions = np.arange(20, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, positions, crs=32633)
        points.ds.index = np.tile(["z", "a"], 10)
        original = points.ds.copy(deep=True)
        result = points.cosample(points, subsample=5, random_state=42)
        repeated = points.cosample(points, subsample=5, random_state=42)

        # Require reproducible selection in native point order
        assert len(result.ds) == 5
        assert np.all(np.diff(result.ds["self"]) > 0)
        assert_geodataframe_equal(result.ds, repeated.ds)

        # Editing a standard spatial result must not modify the source observations
        result.ds["self"] = -1
        assert_geodataframe_equal(points.ds, original)


class TestValidation:
    """Invalid output supports and ambiguous value names."""

    @pytest.mark.parametrize("name", ["self", "other", "geometry"])
    def test_reserved_auxiliary_names(self, name: str) -> None:
        """Checks that auxiliary names cannot replace a primary value or the point geometry column."""

        # Use an otherwise valid comparison to isolate output name validation
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        with pytest.raises(ValueError, match="cannot be"):
            raster.cosample(raster, auxiliary={name: raster})

    def test_conflicting_raster_point_mode(self) -> None:
        """Checks that an explicit point resampling mode rejects a raster output target."""

        # Request point output while explicitly selecting the raster's grid
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        points = raster.to_pointcloud()
        with pytest.raises(ValueError, match="conflicts"):
            raster.cosample(points, at="self", raster_point_mode="resample_raster")

    def test_empty_common_support(self) -> None:
        """Checks that an empty common sample raises before creating an unusable spatial output."""

        # Remove all eligible cells or points through the explicit mask
        raster = _raster(np.arange(12, dtype=float).reshape(3, 4))
        points = raster.to_pointcloud()
        with pytest.raises(ValueError, match="no finite data common"):
            raster.cosample(raster, mask=np.zeros(raster.shape, dtype=bool))
        with pytest.raises(ValueError, match="no finite data common"):
            points.cosample(points, mask=np.zeros(len(points.ds), dtype=bool))
