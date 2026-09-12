"""Tests for defining a common geospatial support and reprojecting point/raster values on it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig
from geoutils.sampling.support import (
    _mask_at_support,
    _sampling_support,
    _values_at_support,
)


def _raster(data: NDArrayNum, *, x_origin: float = 0) -> gu.Raster:
    """Simplify creating a raster for the tests below."""

    return gu.Raster.from_array(data, transform=from_origin(x_origin, data.shape[-2], 1, 1), crs=32633, nodata=-99999)


class TestSupport:
    """
    Test the functions to choose a common support and reproject point/raster on them for eager inputs.

    Dask inputs are tested further below in TestSupportChunked, while Multiproc integration is covered in
    test_cosampling.py.

    Here, we specifically test:
    - _sampling_support() chooses point locations by default, or an explicitly requested raster grid.
    - _values_at_support() aligns rasters and reads raster bands, arrays, or point columns at that support.
    - _mask_at_support() places vector, raster, or array masks on raster grids and point locations.
    """

    def test_sampling_support__default_and_explicit(self) -> None:
        """Checks that points take precedence by default and an explicit raster takes precedence when requested."""

        # Create a raster and points on three of its cells
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        x, y = raster.ij2xy(np.array([0, 1, 2]), np.array([1, 2, 3]))
        points = gu.PointCloud.from_xyz(x, y, np.arange(3, dtype=float), crs=raster.crs)

        # Choose the common locations with and without an explicit request
        point_support = _sampling_support((raster, points), None)
        raster_support = _sampling_support((points, raster), raster)

        # Selected support should be the original object
        assert point_support is points
        assert raster_support is raster

    def test_sampling_support__error_array(self) -> None:
        """Checks that an array cannot define spatial locations by itself."""

        # Pass an array without a raster grid or point coordinates
        values = np.arange(6).reshape(2, 3)

        # Using it as support should raise an error
        with pytest.raises(TypeError, match="must select raster or point cloud support"):
            _sampling_support((values,), None)

    def test_values_at_support__raster_grid_alignment(self) -> None:
        """Checks that a shifted raster requires alignment and then returns values on the selected grid."""

        # Shift the value raster by one cell from the selected raster grid
        support = _raster(np.zeros((3, 4), dtype=float))
        source_values = np.arange(12, dtype=float).reshape(3, 4)
        source = _raster(source_values, x_origin=1)

        # Require explicit permission before moving values to the selected grid
        with pytest.raises(ValueError, match="does not share"):
            _values_at_support(
                source,
                None,
                input_support=source,
                support=support,
                support_dataframe=None,
                name="source",
                interpolation="nearest",
                align="raise",
                mp_config=None,
            )
        result = _values_at_support(
            source,
            None,
            input_support=source,
            support=support,
            support_dataframe=None,
            name="source",
            interpolation="nearest",
            align="reproject",
            mp_config=None,
        )

        # Check the three overlapping columns and the nodata values outside the source grid
        expected = np.full(support.shape, np.nan)
        expected[:, 1:] = source_values[:, :3]
        assert np.array_equal(np.ma.filled(result, np.nan), expected, equal_nan=True)

    def test_values_at_support__selected_raster_band(self) -> None:
        """Checks that a raster band selector returns the requested values on the same grid."""

        # Give both raster bands distinct values on the same grid
        base = np.arange(20, dtype=float).reshape(4, 5)
        raster = _raster(np.stack((base, base + 100)))

        # Read the second band on the raster support
        result = _values_at_support(
            raster,
            2,
            input_support=raster,
            support=raster,
            support_dataframe=None,
            name="values",
            interpolation="nearest",
            align="raise",
            mp_config=None,
        )

        # The selected values should come from the second band
        assert np.array_equal(result, base + 100)

    @pytest.mark.parametrize("input_type", ["numpy", "xarray"])
    def test_values_at_support__raster_array(self, input_type: str) -> None:
        """Checks that an array uses its raster input grid and preserves masked cells."""

        # Create an independent array on the raster grid with one nodata value
        raster = _raster(np.ones((4, 5), dtype=float))
        values = np.ma.array(np.arange(20, dtype=float).reshape(4, 5), mask=False)
        values.mask[1, 2] = True
        source: Any = values
        if input_type == "xarray":
            source = xr.DataArray(values.filled(np.nan), dims=("y", "x"))

        # Place the values directly on the grid supplied by their raster input
        result = _values_at_support(
            source,
            None,
            input_support=raster,
            support=raster,
            support_dataframe=None,
            name="values",
            interpolation="nearest",
            align="raise",
            mp_config=None,
        )

        # Values should be exactly equal
        assert np.array_equal(result, values.filled(np.nan), equal_nan=True)

    def test_values_at_support__raster_at_points(self) -> None:
        """Checks that raster values are read at the selected point locations."""

        # Choose point locations at three known raster cells
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        rows, columns = np.array([1, 2, 3]), np.array([1, 2, 3])
        x, y = raster.ij2xy(rows, columns)
        support = gu.PointCloud.from_xyz(x, y, np.zeros(3), crs=raster.crs)

        # Interpolate the raster at the point support
        result = _values_at_support(
            raster,
            None,
            input_support=raster,
            support=support,
            support_dataframe=support.ds,
            name="values",
            interpolation="nearest",
            align="raise",
            mp_config=None,
        )

        # Extract the same cells directly from the raster to verify the interpolated values
        expected = raster.data.filled(np.nan)[rows, columns]
        assert np.array_equal(result, expected)

    def test_values_at_support__point_column(self) -> None:
        """Checks that a selected point column is returned in the original point order."""

        # Create point values with duplicate row labels and a separate numeric column
        positions = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, positions + 10, crs=32633)
        points.ds["weight"] = 2 * positions
        points.ds.index = ["a", "b", "a", "c", "b"]

        # Read the selected column at the same ordered point locations
        result = _values_at_support(
            points,
            "weight",
            input_support=points,
            support=points,
            support_dataframe=points.ds,
            name="weight",
            interpolation="nearest",
            align="raise",
            mp_config=None,
        )

        # The returned array should follow dataframe row order (even when several rows have the same index label)
        expected = points.ds["weight"].to_numpy()
        assert np.array_equal(result, expected)

    def test_mask_at_support__vector_on_raster(self) -> None:
        """Checks that a vector mask properly rasterizes inside its geometry."""

        # Mask the first two columns of a raster grid with one polygon
        raster = _raster(np.arange(20, dtype=float).reshape(4, 5))
        geometry = gpd.GeoDataFrame(geometry=[box(0, 0, 2, 4)], crs=raster.crs)

        # Place the vector mask on the raster support
        result = _mask_at_support(geometry, raster)
        assert result is not None

        # Check that only cells inside the polygon are masked
        expected = np.zeros(raster.shape, dtype=bool)
        expected[:, :2] = True
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize("mask_mode", ["inside", "outside"])
    def test_mask_at_support__vector_on_points(self, mask_mode: str) -> None:
        """Checks that a vector input (without input feature) perform inside/outside geometry masking."""

        # Place two points inside a polygon and two points outside it
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        rows, columns = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
        x, y = raster.ij2xy(rows, columns)
        points = gu.PointCloud.from_xyz(x, y, np.arange(4, dtype=float), crs=raster.crs)
        mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 3, 3, 6)], crs=raster.crs))

        # Create the polygon mask on the point support with inside/outside mode
        result = _mask_at_support(mask, points, support_dataframe=points.ds, mask_mode=mask_mode)
        assert result is not None

        # Check exact boolean output is as expected
        expected = np.array([True, True, False, False])
        if mask_mode == "outside":
            expected = ~expected
        assert np.array_equal(result, expected)

    def test_mask_at_support__raster_on_points(self) -> None:
        """Checks that a raster mask selects points in true cells and excludes points outside its grid."""

        # Set the first three raster columns to true and add one point beyond the raster
        raster = _raster(np.arange(30, dtype=float).reshape(5, 6))
        allowed = np.indices(raster.shape)[1] < 3
        mask = gu.Raster.from_array(allowed, raster.transform, raster.crs)
        rows, columns = np.array([0, 1, 3, 4]), np.array([1, 2, 4, 5])
        x, y = raster.ij2xy(rows, columns)
        x, y = np.append(x, 100.0), np.append(y, 100.0)
        points = gu.PointCloud.from_xyz(x, y, np.arange(5, dtype=float), crs=raster.crs)

        # Read the boolean raster at each point location
        result = _mask_at_support(mask, points, support_dataframe=points.ds)
        assert result is not None

        # Check true cells, false cells, and the point outside the raster
        assert np.array_equal(result, [True, True, False, False, False])

    def test_mask_at_support__masked_point_array(self) -> None:
        """Checks that nodata and false entries in a point mask are both excluded."""

        # Give one point a nodata mask value and another point a false value
        positions = np.arange(5, dtype=float)
        points = gu.PointCloud.from_xyz(positions, positions, positions, crs=32633)
        mask = np.ma.array([True, True, True, False, True], mask=[False, False, True, False, False])

        # Place the plain mask array on the ordered point support
        result = _mask_at_support(mask, points, support_dataframe=points.ds)
        assert result is not None

        # Check that only true, finite mask entries remain selected
        assert np.array_equal(result, [True, True, False, False, True])


class TestSupportChunked:
    """
    Test module checking that the support helpers respect Dask/MP chunked execution and that their values exactly
    match eager results.
    """

    def test_values_at_support__backends(self, tmp_path: Path) -> None:
        """Checks that Dask and Multiproc raster alignment exactly matches eager without loading the inputs."""

        import_optional("dask")
        import dask.array as da

        # Create a two-band raster with one nodata cell, and shift the selected support by one column
        values = np.arange(30, dtype=float).reshape(5, 6)
        source = _raster(np.stack((values + 100, values)))
        source.data[1, 2, 3] = np.ma.masked
        support = _raster(np.zeros(source.shape, dtype=float), x_origin=1)

        # Calculate the eager reference from the selected second band
        expected = _values_at_support(
            source,
            2,
            input_support=source,
            support=support,
            support_dataframe=None,
            name="values",
            interpolation="nearest",
            align="reproject",
            mp_config=None,
        )

        # Reproject the same values through Dask chunks and Multiproc tiles
        lazy_source = source.to_xarray().chunk({"band": 1, "y": 2, "x": 3})
        dask_result = _values_at_support(
            lazy_source,
            2,
            input_support=lazy_source,
            support=support,
            support_dataframe=None,
            name="values",
            interpolation="nearest",
            align="reproject",
            mp_config=None,
        )
        source_file = tmp_path / "values.tif"
        source.to_file(source_file)
        multiproc_source = gu.Raster(source_file, load_data=False)
        outfile = tmp_path / "aligned_values.tif"
        multiproc_result = _values_at_support(
            multiproc_source,
            2,
            input_support=multiproc_source,
            support=support,
            support_dataframe=None,
            name="values",
            interpolation="nearest",
            align="reproject",
            mp_config=MultiprocConfig(chunks=(2, 3), outfile=str(outfile)),
        )

        # Check Dask laziness and the file loading state before comparing every output value
        assert isinstance(lazy_source.data, da.Array)
        assert isinstance(dask_result, da.Array)
        assert not multiproc_source.is_loaded
        assert outfile.exists()
        expected_values = np.ma.filled(expected, np.nan)
        assert np.array_equal(dask_result.compute(), expected_values, equal_nan=True)
        assert np.array_equal(np.ma.filled(multiproc_result, np.nan), expected_values, equal_nan=True)
        assert isinstance(lazy_source.data, da.Array)
        assert not multiproc_source.is_loaded

    def test_mask_at_support__backends(self, tmp_path: Path) -> None:
        """Checks that Dask and Multiproc vector masks exactly match eager without loading the support grids."""

        import_optional("dask")
        import dask.array as da

        # Create a raster support and a polygon that covers its first three columns
        support = _raster(np.zeros((5, 6), dtype=float))
        mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 3, 5)], crs=support.crs))

        # Calculate the eager reference, then place the same vector mask through Dask chunks
        expected = _mask_at_support(mask, support, align="reproject")
        lazy_support = support.to_xarray().chunk({"y": 2, "x": 3})
        dask_result = _mask_at_support(mask, lazy_support.rst, align="reproject")

        # Write the support grid to disk and place the vector mask through Multiproc tiles
        support_file = tmp_path / "support.tif"
        support.to_file(support_file)
        multiproc_support = gu.Raster(support_file, load_data=False)
        outfile = tmp_path / "aligned_mask.tif"
        multiproc_result = _mask_at_support(
            mask,
            multiproc_support,
            align="reproject",
            mp_config=MultiprocConfig(chunks=(2, 3), outfile=str(outfile)),
        )

        # Check Dask laziness and the file loading state before comparing every output mask value
        assert expected is not None
        assert dask_result is not None and isinstance(dask_result, da.Array)
        assert multiproc_result is not None
        assert isinstance(lazy_support.data, da.Array)
        assert not multiproc_support.is_loaded
        assert outfile.exists()
        assert np.array_equal(dask_result.compute(), expected)
        assert np.array_equal(multiproc_result, expected)
        assert isinstance(lazy_support.data, da.Array)
        assert not multiproc_support.is_loaded
