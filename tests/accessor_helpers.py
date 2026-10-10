"""Shared fixtures and output comparisons for class and accessor tests."""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from pyproj import CRS

from geoutils import ErrorStructure, PointCloud, Variogram, Vector
from geoutils._dispatch import _get_raster_interface


@pytest.fixture
def mixed_dataset() -> Any:
    """A mixed Xarray Dataset for tests: rasters, point values and a scalar on independent dimensions."""

    import xarray as xr
    from rasterio.transform import from_origin

    from geoutils import DataArrayPointCloudAccessor, DataArrayRasterAccessor

    # We use uneven raster/point shapes to test shorter final Dask chunks
    values = np.arange(35, dtype=float).reshape(5, 7)
    dem = DataArrayRasterAccessor.from_array(values, from_origin(0, 5, 1, 1), 32631, nodata=-9999)
    slope = dem.copy(data=values / 10)
    x = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 0.5, 1.5])
    y = np.array([4.5, 3.5, 2.5, 1.5, 0.5, 4.5, 3.5, 2.5, 1.5])
    points = DataArrayPointCloudAccessor.from_xyz(x, y, np.arange(9, dtype=float), 32631)
    points = points.rename({"x": "x_point", "y": "y_point"})
    sigma = points.copy(data=np.linspace(1, 2, 9))

    # We use independent point coordinate names to avoid collision with the raster X/Y axes
    dataset = xr.Dataset(
        {
            "dem": dem,
            "slope": slope,
            "point_z": points,
            "point_sigma": sigma,
            "acquisition_date": xr.DataArray(np.datetime64("2026-01-01")),
        },
        attrs={"title": "mixed spatial data"},
    )
    return dataset


def assert_output_equal(output1: Any, output2: Any, use_allclose: bool = False, strict_masked: bool = True) -> None:
    """
    Return equality of different output types, shared by all test modules comparing consistency across data
    structures (most test_base across all object types, also test_accessors, etc).
    """

    # For point clouds, accepting accessor-backed GeoDataFrames
    if isinstance(output1, PointCloud) or isinstance(output2, PointCloud):
        if not isinstance(output1, PointCloud):
            output1, output2 = output2, output1
        if isinstance(output2, gpd.GeoDataFrame):
            output2 = PointCloud(output2, data_name=output1.data_name)
        assert output1.pointcloud_equal(output2)

    # For vectors, accepting accessor-backed GeoDataFrames
    elif isinstance(output1, Vector) or isinstance(output2, Vector):
        vector1 = output1 if isinstance(output1, Vector) else Vector(output1)
        vector2 = output2 if isinstance(output2, Vector) else Vector(output2)
        assert vector1.vector_equal(vector2)

    # For two raster: Xarray or Raster objects
    elif (raster := _get_raster_interface(output1)) is not None:
        if use_allclose:
            assert raster.raster_allclose(output2, warn_failure_reason=True, strict_masked=strict_masked)
        else:
            assert raster.raster_equal(output2, warn_failure_reason=True, strict_masked=strict_masked)

    # For arrays
    elif isinstance(output1, np.ndarray):
        if np.ma.isMaskedArray(output1):
            output1 = output1.filled(np.nan)
        if np.ma.isMaskedArray(output2):
            output2 = output2.filled(np.nan)
        if use_allclose:
            assert np.allclose(output1, np.asarray(output2), equal_nan=True)
        else:
            assert np.array_equal(output1, np.asarray(output2), equal_nan=True)

    # For tuple of arrays
    elif isinstance(output1, tuple) and isinstance(output1[0], np.ndarray):
        assert np.array_equal(np.array(output1), np.array([np.asarray(value) for value in output2]), equal_nan=True)

    # For a dictionary of numeric values
    elif isinstance(output1, dict):
        df1 = pd.DataFrame(index=[0], data=output1)
        df2 = pd.DataFrame(index=[0], data=output2)
        assert_frame_equal(df1, df2, check_dtype=False)

    # For GeoPandas objects
    elif isinstance(output1, gpd.GeoDataFrame):
        if isinstance(output2, xr.DataArray):
            output2 = output2.pc.to_geoutils().gdf
        assert_geodataframe_equal(output1, output2)
    elif isinstance(output1, gpd.GeoSeries):
        assert_geoseries_equal(output1, output2)

    # For tabular statistics
    elif isinstance(output1, pd.DataFrame):
        assert_frame_equal(output1, output2)
    elif isinstance(output1, pd.Series):
        assert_series_equal(output1, output2)
    elif isinstance(output1, pd.Index):
        assert output1.equals(output2)

    # For fitted error structures
    elif isinstance(output1, ErrorStructure):
        assert isinstance(output2, ErrorStructure)
        assert output1.components == output2.components
        assert output1.empirical_variogram == output2.empirical_variogram

    # For lightweight variogram records
    elif isinstance(output1, Variogram):
        assert isinstance(output2, Variogram)
        assert np.allclose(output1.lags, output2.lags)
        assert np.allclose(output1.semivariance, output2.semivariance, equal_nan=True)
        assert np.array_equal(output1.counts, output2.counts)
        assert output1.model == output2.model

    # For labelled pair samples
    elif isinstance(output1, (xr.Dataset, xr.DataArray)):
        if isinstance(output1, xr.Dataset) and "crs" in output1.attrs:
            # Class and accessor pairs can describe the same CRS with different WKT strings
            crs1, crs2 = output1.attrs["crs"], output2.attrs["crs"]
            if crs1 is None or crs2 is None:
                assert crs1 is crs2
            else:
                assert CRS.from_user_input(crs1).equals(CRS.from_user_input(crs2), ignore_axis_order=True)
            output2 = output2.copy()
            output2.attrs["crs"] = crs1
        xr.testing.assert_identical(output1, output2)

    # For any other object type
    else:
        assert output1 == output2


def assert_xarray_equal(actual: xr.Dataset, expected: xr.Dataset) -> None:
    """Compare complete Xarray Datasets, including dtype."""

    # Values, dimensions, coordinates, names and attributes must agree
    assert isinstance(actual, xr.Dataset)
    xr.testing.assert_identical(actual, expected)
    # And dtype as well (can be missed even with same values)
    for name in expected.variables:
        assert actual[name].dtype == expected[name].dtype


def assert_dataset_output_equal(
    source: xr.Dataset, actual: xr.Dataset, outputs: dict[str, xr.DataArray], *, rtol: float = 0
) -> None:
    """Compare multiple DataArray outputs with a rebuilt Dataset, to facilitate tests of Dataset accessors."""

    # We remove replaced axes before merging outputs, so Xarray cannot align them to the old grid
    independent = source.drop_vars(list(outputs))
    used_dimensions = {dimension for variable in independent.data_vars.values() for dimension in variable.dims}
    output_dimensions = {dimension for variable in outputs.values() for dimension in variable.dims}
    replaced_dimensions = output_dimensions.intersection(independent.dims).difference(used_dimensions)
    independent = independent.drop_dims(replaced_dimensions)

    # Output coordinates describe changed grids; independent values and Dataset attributes come from the source
    expected = xr.merge([xr.Dataset(outputs), independent], compat="override")
    expected.attrs = source.attrs

    # An untouched raster still uses its original CRS description, even if an equivalent output WKT differs
    mappings = {variable.encoding.get("grid_mapping") for variable in independent.data_vars.values()}
    expected = expected.assign_coords({name: independent[name] for name in mappings if name in independent.coords})

    # Compare floating values within tolerance, then compare metadata and untouched variables exactly
    comparison = actual.copy(deep=False) if rtol else actual
    if rtol:
        for name in outputs:
            assert actual[name].dtype == expected[name].dtype
            np.testing.assert_allclose(actual[name].data, expected[name].data, rtol=rtol, atol=0)
            comparison[name].data = expected[name].data
    assert_xarray_equal(comparison, expected)


def assert_variograms_equal(actual: dict[str, Variogram], expected: dict[str, Variogram]) -> None:
    """Compare dictionaries of variograms, including all their arrays and metadata."""

    # Dictionary order records the selected variable order
    assert isinstance(actual, dict)
    assert list(actual) == list(expected)

    # Xarray exposes every variogram value and its estimation metadata
    for name, reference in expected.items():
        assert isinstance(actual[name], Variogram)
        assert_xarray_equal(actual[name].to_xarray(), reference.to_xarray())
