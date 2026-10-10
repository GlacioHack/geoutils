"""Tests on Xarray accessor mirroring Raster API."""

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils import examples, open_raster
from geoutils._misc import import_optional
from geoutils.raster.base import RasterBase
from tests.accessor_helpers import (
    assert_dataset_output_equal,
    assert_variograms_equal,
)
from tests.accessor_helpers import (
    mixed_dataset as mixed_dataset,
)


class TestDataArrayRasterAccessor:
    """
    Test for Xarray DataArray accessor subclass.

    Note: This test class only tests functionalities that are specific to the DataArrayRasterAccessor subclass.
    Overridden abstract methods, loading behaviour and Dask laziness are tested in test_base directly to mirror
    Raster tests.

    This class thus tests:
    - The open_raster function,
    - The instantiation __init__ through ds.rst,
    - The to_geoutils() method.
    """

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("as_type", [None, "dataarray", "dataset"])
    @pytest.mark.parametrize("data_name", [None, "elevation"])
    @pytest.mark.parametrize("bands", [1, 2])
    @pytest.mark.parametrize("is_mask", [False, True])
    def test_open_raster__return_type(
        self,
        tmp_path: Path,
        as_type: Literal["dataarray", "dataset"] | None,
        data_name: str | None,
        bands: int,
        is_mask: bool,
    ) -> None:
        """Checks that opening returns the requested type and name with exact values, bands and georeferencing."""

        # We write 1/2 bands with a NaN to check masking/nodata behaviour
        values = np.arange(bands * 35, dtype=np.float32).reshape(bands, 5, 7)
        values[0, 2, 3] = np.nan
        transform = from_origin(500000, 8600000, 20, 20)
        reference = gu.Raster.from_array(values, transform, 32633, nodata=-9999)
        path = tmp_path / "opening.tif"
        reference.to_file(path)

        # Check we open with the right type
        if as_type is None:
            result = open_raster(str(path), data_name=data_name, is_mask=is_mask)
        else:
            result = open_raster(str(path), as_type=as_type, data_name=data_name, is_mask=is_mask)
        if as_type == "dataset":
            assert isinstance(result, xr.Dataset)
            name = data_name or "raster"
            assert result.rst.variables == [name]
            variable = result[name]
        else:
            assert isinstance(result, xr.DataArray)
            variable = result
            if data_name is not None:
                assert variable.name == data_name
        assert variable._in_memory is is_mask

        # Then we check proper metadata attribute, and equal data
        expected = values[0] if bands == 1 else values
        if is_mask:
            expected = expected.astype(bool)
        else:
            assert variable.rio.encoded_nodata == reference.nodata
        assert variable.dims == (("y", "x") if bands == 1 else ("band", "y", "x"))
        assert variable.rio.crs == reference.crs
        assert variable.rio.transform() == transform
        np.testing.assert_array_equal(variable.data, expected)

    @pytest.mark.parametrize("bands", [1, 2])
    def test_from_array__rotated_grid_coordinates(self, bands: int) -> None:
        """Checks that rotated grids store each pixel's map coordinates and affine transform."""

        # Rotation makes the X and Y coordinates depend on both row and column
        values = np.arange(bands * 6).reshape((bands, 2, 3))
        transform = Affine(2, 0.5, 10, 0.25, -2, 20)

        # Construct a single-band or multi-band xarray raster from the same grid
        result = gu.DataArrayRasterAccessor.from_array(values, transform=transform, crs=32606)
        rows, columns = np.indices((2, 3))
        expected_x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
        expected_y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f

        # Check map coordinates, georeferencing, and original values
        assert result.rst.transform == transform
        assert result.rst.crs.to_epsg() == 32606
        np.testing.assert_allclose(result.coords["xc"], expected_x)
        np.testing.assert_allclose(result.coords["yc"], expected_y)
        np.testing.assert_array_equal(result.data, values[0] if bands == 1 else values)

    @pytest.mark.parametrize("shape", [(1, 3), (3, 1), (1, 1)])
    @pytest.mark.parametrize("bands", [1, 2])
    def test_open_raster__single_row_or_column(self, tmp_path: Path, shape: tuple[int, int], bands: int) -> None:
        """Checks that opening a raster returns spatial dimensions of length one and every requested band."""

        # A single row or column is still a two-dimensional grid, including for several bands
        values = np.arange(bands * np.prod(shape), dtype=np.float32).reshape((bands, *shape))
        path = tmp_path / "narrow.tif"
        transform = from_origin(0, 3, 1, 1)
        gu.Raster.from_array(values, transform, 32631).to_file(path)

        # Opening removes only a single band dimension, while retaining the grid coordinates and values
        result = open_raster(str(path))
        expected = values[0] if bands == 1 else values
        assert result.dims == (("y", "x") if bands == 1 else ("band", "y", "x"))
        assert result.rst.shape == shape
        assert result.rst.transform == transform
        np.testing.assert_array_equal(result.data, expected)

    @pytest.mark.parametrize("as_type", ["dataarray", "dataset"])
    def test_open_raster__downsample(self, tmp_path: Path, as_type: Literal["dataarray", "dataset"]) -> None:
        """Checks that downsampling uses a suitable overview while keeping the requested output grid."""

        # Write input for full array, then open with downsampling (without overview)
        values = np.arange(5_000, dtype=np.int32).reshape(50, 100)
        path = tmp_path / "gradient.tif"
        gu.Raster.from_array(values, from_origin(0, 50, 1, 1), 4326).to_file(path)
        without_overview = open_raster(str(path), downsample=6, as_type=as_type)
        without_overview = without_overview["raster"] if as_type == "dataset" else without_overview
        without_overview_values = without_overview.values.copy()

        # Add overview for factors of 2 and 4, factor 4 should be the closest suitable overview
        with rio.open(path, "r+") as dataset:
            dataset.build_overviews([2, 4], rio.enums.Resampling.nearest)

        # Check that downsampling the new file uses the overview (slightly changes the sampled values)
        # without changing the size/transform of the requested grid
        result = open_raster(str(path), downsample=6, as_type=as_type)
        result = result["raster"] if as_type == "dataset" else result
        assert not result._in_memory
        output = result.values
        expected = gu.Raster(path, downsample=6).data.data

        assert result.rst.shape == (8, 16)
        assert result.rst.transform == from_origin(0, 50, 6, 6)
        np.testing.assert_array_equal(output, expected)
        assert not np.array_equal(output, without_overview_values)

    @pytest.mark.parametrize("as_type", ["dataarray", "dataset"])
    def test_open_raster__overview_level(self, tmp_path: Path, as_type: Literal["dataarray", "dataset"]) -> None:
        """Checks that overview selection is distinct from downsampling."""

        # Store two overview levels (the level 0 represents the factor-2 grid)
        values = np.arange(16 * 16, dtype=np.int16).reshape(16, 16)
        path = tmp_path / "overviews.tif"
        gu.Raster.from_array(values, from_origin(0, 16, 1, 1), 4326).to_file(path)
        with rio.open(path, "r+") as dataset:
            dataset.build_overviews([2, 4], rio.enums.Resampling.nearest)

        # Check error is raised when combining exact overview selection with downsampling
        overview = open_raster(str(path), overview_level=0, as_type=as_type)
        overview = overview["raster"] if as_type == "dataset" else overview
        assert overview.rst.shape == (8, 8)
        assert overview.rst.transform == from_origin(0, 16, 2, 2)
        with pytest.raises(ValueError, match="downsample and overview_level cannot be used together"):
            open_raster(str(path), downsample=3, overview_level=0, as_type=as_type)

    @pytest.mark.parametrize("as_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("downsample", [1, 6])
    def test_open_raster__downsample_loading_laziness(
        self, tmp_path: Path, as_type: Literal["dataarray", "dataset"], downsample: int
    ) -> None:
        """Checks that chunked downsampling stays lazy and exactly matches eager opening."""

        pytest.importorskip("dask.array")
        from dask.callbacks import Callback

        # Write input on dimensions not divisible by 6
        values = np.arange(5_000, dtype=np.int32).reshape(50, 100)
        path = tmp_path / "gradient.tif"
        gu.Raster.from_array(values, from_origin(0, 50, 1, 1), 4326).to_file(path)

        # Uneven tiles expose shorter final chunks on both the native and reduced grids
        eager = open_raster(str(path), downsample=downsample, as_type=as_type)
        with Callback(pretask=lambda *args: pytest.fail("Opening rasters must read only metadata")):
            lazy = open_raster(str(path), downsample=downsample, chunks={"y": 3, "x": 7}, as_type=as_type)
        eager_values = eager["raster"] if as_type == "dataset" else eager
        lazy_values = lazy["raster"] if as_type == "dataset" else lazy
        assert not eager_values._in_memory
        assert not lazy_values._in_memory
        expected_chunks = ((3,) * 16 + (2,), (7,) * 14 + (2,)) if downsample == 1 else ((3, 3, 2), (7, 7, 2))
        assert lazy_values.chunks == expected_chunks

        # Check result leaves source lazy and matches exactly with eager resampling
        computed = lazy.compute()
        xr.testing.assert_identical(computed, eager)
        computed_values = computed["raster"] if as_type == "dataset" else computed
        assert computed_values._in_memory
        assert not lazy_values._in_memory

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_copy(self, path_raster: str) -> None:

        ds = open_raster(path_raster)
        ds_copy = ds.rst.copy()

        assert np.array_equal(ds.data, ds_copy.data, equal_nan=True)
        assert ds.rst.transform == ds_copy.rst.transform
        assert ds.rst.crs == ds_copy.rst.crs
        assert ds.rst.nodata == ds_copy.rst.nodata

    @pytest.mark.parametrize("lazy", [False, True])
    def test_to_geoutils__loading_laziness(self, tmp_path: Path, lazy: bool) -> None:
        """Checks that native conversion loads exact values while keeping a Dask source lazy."""

        # Write a test file with a missing pixel and Point metadata
        values = np.arange(35, dtype=np.float32).reshape(5, 7)
        values[2, 3] = np.nan
        reference = gu.Raster.from_array(
            values, from_origin(500000, 8600000, 20, 20), 32633, nodata=-9999, area_or_point="Point"
        )
        path = tmp_path / "conversion.tif"
        reference.to_file(path)
        if lazy:
            pytest.importorskip("dask.array")
        source = open_raster(str(path), chunks={"y": 3, "x": 4} if lazy else None)
        graph = source.data if lazy else None
        assert not source._in_memory

        # Convert to a loaded Raster while keeping the caller's Dask array lazy
        result = source.rst.to_geoutils()
        assert isinstance(result, gu.Raster) and result.is_loaded
        assert source._in_memory is not lazy
        if lazy:
            assert source.data is graph

        # Check exact values, missing pixels and the complete spatial reference
        assert reference.raster_equal(result, strict_masked=False, warn_failure_reason=True)
        if lazy:
            assert source.data is graph and not source._in_memory

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_open__loaded(self, path_raster: str) -> None:
        """
        Test that a DataArray opened using "open_raster" maintains implicit loading logic.

        Tests checking loading for all attributes and methods are done in TestBase.

        Note: this is different from using lazy Dask arrays: for any array type, Xarray only loads metadata, and
        implicitly loads data in memory when .data or .load() is called.
        """

        # Open raster with/without chunks, should not load in memory yet
        ds = open_raster(path_raster)
        assert not ds._in_memory

        # The array should be NumPy
        assert isinstance(ds.data, np.ndarray)
        ds.load()
        assert ds._in_memory

    @pytest.mark.parametrize("path_raster", [landsat_b4_path, aster_dem_path])
    def test_open__dask(self, path_raster: str) -> None:
        """
        Check that a DataArray opened with chunks using "open_raster" maintains Dask laziness.

        Note: this is different from loading mechanism of Xarray (triggers when calling .data).
        """
        pytest.importorskip("dask")
        import dask.array as da

        # Open raster lazily with chunks
        ds = open_raster(path_raster, chunks={"band": 1, "x": 10, "y": 10})

        # Array should be a Dask array (chunks exist)
        ds_arr = ds.data
        assert not ds._in_memory
        assert isinstance(ds_arr, da.Array)
        assert ds_arr.chunks is not None

        # After compute, it should be a NumPy array
        ds_comp = ds.compute()
        assert isinstance(ds_comp.data, np.ndarray)
        assert ds_comp._in_memory

    def test_equality__dask_reduces_lazily(self) -> None:
        """Compare Dask raster data through scalar reductions without loading the DataArray."""

        pytest.importorskip("dask")
        import dask.array as da

        array = np.arange(20, dtype=np.float32).reshape(4, 5)
        raster = gu.Raster.from_array(array, transform=from_origin(0, 4, 1, 1), crs=4326, nodata=None)
        lazy = gu.DataArrayRasterAccessor.from_array(
            da.from_array(array, chunks=(2, 3)), transform=raster.transform, crs=raster.crs, nodata=None
        )
        close = lazy.copy(data=lazy.data + 1e-7)
        changed = lazy.copy(data=lazy.data + 1)

        assert raster.raster_equal(lazy)
        assert lazy.rst.raster_equal(raster)
        assert raster.raster_allclose(close, atol=1e-6)
        assert not raster.raster_equal(close)
        assert not raster.raster_allclose(changed)
        assert not lazy._in_memory

    def test_open__dask_nodata_can_be_written(self, tmp_path: Path) -> None:
        """Write a lazily opened nodata raster without conflicting xarray metadata."""

        pytest.importorskip("dask")

        # Opening a masked file moves its encoded nodata to one unambiguous attribute
        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})
        assert ds.rst.nodata is not None
        assert "_FillValue" not in ds.encoding

        # The final writer must preserve nodata while computing the Dask array
        output_file = tmp_path / "dask-nodata.tif"
        ds.rst.to_file(output_file)
        with rio.open(output_file) as output:
            assert output.nodata == ds.rst.nodata

    def test_cross_type_outputs_are_accessors(self) -> None:
        """Return accessor-backed vectors and point clouds when a raster operation changes type."""

        # Create one compact raster shared by every cross-type operation
        ds = gu.DataArrayRasterAccessor.from_array(
            data=np.array([[1, 1], [0, 0]], dtype=np.uint8),
            transform=from_origin(0, 2, 1, 1),
            crs=4326,
            nodata=None,
        )

        # Polygonization returns a GeoDataFrame with the vector accessor
        vector = ds.rst.polygonize(target_values=1)
        assert isinstance(vector, gpd.GeoDataFrame)
        assert vector.vct.to_geoutils().vector_equal(gu.Vector(vector))

        # Raster-to-point operations return GeoDataFrames with point-cloud metadata
        pointcloud = ds.rst.to_pointcloud(skip_nodata=False)
        assert isinstance(pointcloud, gpd.GeoDataFrame)
        assert pointcloud.pc.data_name == "b1"

        interpolated = ds.rst.interp_at_points((np.array([0.5]), np.array([1.5])), method="nearest")
        assert isinstance(interpolated, gpd.GeoDataFrame)
        assert interpolated.pc.data_name == "z"

        # Geometric footprint helpers also retain the vector accessor
        footprint = ds.rst.get_footprint_projected(ds.rst.crs)
        assert isinstance(footprint, gpd.GeoDataFrame)

    def test_stats__dask_global_quantile_stats(self) -> None:
        """Checks that global quantile statistics match eager results without loading the Dask raster."""

        pytest.importorskip("dask")
        import dask.array as da

        # Open the same raster eagerly and in chunks so both calculations use the same pixel values
        base = open_raster(self.aster_dem_path)
        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})

        # Check that the chunked raster starts with lazy data
        assert isinstance(ds.data, da.Array)
        assert not ds._in_memory

        # Compare statistics that need values from all chunks (median, percentiles and related measures)
        for stat in ["median", "90th percentile", "le90", "nmad", "iqr"]:
            expected = base.rst.stats(stat)
            actual = ds.rst.stats(stat)

            # stats() computes its small summary result, but it must not replace the source with loaded data
            assert np.isscalar(actual)
            assert actual == pytest.approx(expected, rel=1e-5)
            assert isinstance(ds.data, da.Array)
            assert not ds._in_memory

    def test_reproject__dask_keeps_dimension_order_for_stats(self) -> None:
        """Checks that reprojected Dask rasters keep valid dimensions and support global statistics."""

        pytest.importorskip("dask")
        import dask.array as da

        # Open one raster in chunks, then reproject its CRS and pixel size along the two affected code paths
        ds = open_raster(self.aster_dem_path, chunks={"band": 1, "x": 100, "y": 100})
        reprojected_crs = ds.rst.reproject(crs=4326)
        reprojected_res = ds.rst.reproject(res=(ds.rst.res[0] * 2, ds.rst.res[1] / 2), resampling="bilinear")

        # Check that both outputs use rioxarray's y/x order and stay lazy after calculating their mean
        for reprojected in [reprojected_crs, reprojected_res]:
            expected = reprojected.compute().rst.stats("mean")
            actual = reprojected.rst.stats("mean")

            assert reprojected.dims == ("y", "x")
            assert isinstance(reprojected.data, da.Array)
            assert not reprojected._in_memory
            assert np.isscalar(actual)
            # Partial sums are combined in a different order across chunks, so allow their small rounding difference
            assert actual == pytest.approx(expected, rel=1e-7)
            assert np.isfinite(actual)

    def test_chunked_rasterize_paths_accept_dask_chunk_tuples(self) -> None:
        """Regression test for xarray/Dask rasterization paths receiving normalized chunk tuples."""

        pytest.importorskip("dask")
        import dask.array as da

        arr = np.zeros((12, 10), dtype=np.uint8)
        arr[2:9, 3:8] = 1
        dask_arr = da.from_array(arr, chunks=(5, 4))
        ds = gu.DataArrayRasterAccessor.from_array(
            data=dask_arr,
            transform=from_origin(0, 12, 1, 1),
            crs=4326,
            nodata=0,
        )
        vector = gu.Vector(gpd.GeoDataFrame({"geometry": [box(2, 4, 9, 11)]}, crs=4326))

        mask = vector.create_mask(ds.rst, dask=True)
        assert isinstance(mask.data, da.Array)
        assert mask.data.chunks == dask_arr.chunks
        assert bool(mask.compute().data[3, 3])

        polygons = ds.rst.polygonize(target_values=1)
        rasterized = polygons.vct.rasterize(ds.rst, in_value=1, out_value=0, out_dtype=np.uint8)
        assert isinstance(rasterized.data, da.Array)
        assert rasterized.data.chunks == dask_arr.chunks
        assert np.array_equal(rasterized.compute().data, arr)

        rasterized_from_dataarray = polygons.vct.rasterize(ds, in_value=1, out_value=0, out_dtype=np.uint8)
        assert isinstance(rasterized_from_dataarray.data, da.Array)
        assert rasterized_from_dataarray.data.chunks == dask_arr.chunks

    @pytest.mark.parametrize("as_type", ["geodataframe", "scene"])
    def test_open_raster__error_invalid_type(self, tmp_path: Path, as_type: str) -> None:
        """Checks an error is raised before reading a file for an unsupported raster return type."""

        path = tmp_path / "absent.tif"
        with pytest.raises(ValueError, match="as_type"):
            open_raster(str(path), as_type=as_type)  # type: ignore[arg-type]


class TestDatasetRasterAccessor:
    """Test module for the Xarray Dataset raster accessor ``rst``."""

    def test_accessor(self, mixed_dataset: xr.Dataset) -> None:
        """
        First, check that raster and point accessors coexist and discover their respective variables.

        We test variable discovery separately because a mixed Dataset needs to distinguish its rasters from its point
        values.
        """

        # Identify representations from dimensions and georeferencing
        assert isinstance(mixed_dataset.rst, gu.DatasetRasterAccessor)
        assert not isinstance(mixed_dataset.rst, RasterBase)
        assert isinstance(mixed_dataset.dem.rst, gu.DataArrayRasterAccessor)
        assert isinstance(mixed_dataset.dem.rst, RasterBase)
        assert mixed_dataset.rst.variables == ["dem", "slope"]
        assert mixed_dataset.pc.variables == ["point_z", "point_sigma"]
        assert not hasattr(mixed_dataset.rst, "ds")

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize("coordinates,with_time", [("native", False), ("cf", False), ("cf", True)])
    @pytest.mark.parametrize(
        "method,options,variables,transposed",
        [
            ("reproject", {"res": 2}, None, False),
            ("crop", {"bbox": (1, 1, 5, 4)}, None, False),
            ("icrop", {"bbox": (1, 1, 5, 4)}, None, False),
            ("translate", {"xoff": 1, "yoff": 2}, None, False),
            ("filter", {"method": "mean", "size": 3}, None, False),
            ("filter", {"method": "mean", "size": 3}, ["dem"], False),
            ("filter", {"method": "median", "size": 3}, None, False),
            ("sieve", {"size": 2}, ["regions"], False),
            ("fill_nodata", {"interpolation": "nearest"}, None, False),
            ("clip", {"mask": (1, 1, 5, 4)}, None, False),
            ("reproject", {"res": 2}, None, True),
            ("filter", {"method": "mean", "size": 3}, ["dem"], True),
        ],
    )
    def test_methods__raster_outputs(
        self,
        mixed_dataset: xr.Dataset,
        method: str,
        options: dict[str, Any],
        transposed: bool,
        variables: list[str] | None,
        coordinates: str,
        with_time: bool,
        lazy: bool,
    ) -> None:
        """
        Checks that Dataset methods producing raster outputs match DataArray calls per recognized raster variable, and
        do not affect independent points and scalar variables in the dataset.

        We test methods with/without selecting a specific variable, with native/CF spatial coordinates and time labels,
        and with/without transposed input, when relevant.

        These combinations cover raster outputs with basic arguments. Other types of outputs (points, dict),
        and changed Dataset behaviour for input arguments are tested separately further below.
        """

        # 1/ Create a test mixed Xarray Dataset by modifying the mixed_dataset fixture
        canonical = mixed_dataset.copy(deep=True)
        if method == "sieve":
            # Add a label raster with one isolated cell that sieve() should merge into the surrounding region
            labels = np.zeros(canonical.dem.shape, dtype=np.int32)
            labels[2, 3] = 1
            canonical = canonical.assign(regions=canonical.dem.copy(data=labels))
        if method == "fill_nodata":
            # Add an interior NaN to both rasters so fill_nodata() has missing values to interpolate
            canonical.dem.data[2, 3] = np.nan
            canonical.slope.data[2, 3] = np.nan

        # Create CF axis attributes to identify spatial dimensions without relying on names
        dimensions = {}
        if coordinates == "cf":
            canonical.x.attrs = {"axis": "X", "units": "m", "valid_range": np.array([0.0, 7.0])}
            canonical.y.attrs = {"axis": "Y", "units": "m"}
            dimensions = {"x": "easting", "y": "northing"}

        # We test a potential time dimension too, to be sure we don't mix time axes when they exist
        if with_time:
            dates = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
            rasters = canonical[canonical.rst.variables].expand_dims(band=dates).copy(deep=True)
            for name in rasters.data_vars:
                rasters[name].data[1] += 100
            canonical = canonical.assign({name: rasters[name] for name in rasters.data_vars})
            dimensions["band"] = "time"
        source = canonical.rename(dimensions)
        if with_time:
            source["weather"] = ("time", [12.0, 14.0])
        if transposed:
            source = source.transpose(dimensions.get("x", "x"), dimensions.get("y", "y"), ...)
        if lazy:
            pytest.importorskip("dask")
            from dask.callbacks import Callback

            # Split rasters into chunks (with shorter final chunks)
            chunks = {dimensions.get("y", "y"): 3, dimensions.get("x", "x"): 4, "point": 4}
            source = source.chunk(chunks)
        original = source.copy(deep=True)

        # 2/ We run the same operation through Dataset and DataArray raster interfaces
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.rst, method)(variables=variables, **options)
        names = source.rst.variables if variables is None else variables
        if lazy:
            # Check that construction is lazy
            assert not tasks
            assert all(source[name].chunks and result[name].chunks for name in names)
            assert result.point_z.data is source.point_z.data
            for name in source.rst.variables:
                if name not in names:
                    assert result[name].data is source[name].data
        outputs = {}
        for name in names:
            # With different dates, separate DataArray calls should be the default
            if with_time:
                slices = [
                    getattr(canonical[name].isel(band=index, drop=True).rst, method)(**options)
                    for index in range(canonical.sizes["band"])
                ]
                output = xr.concat(slices, dim=canonical.band)
            else:
                output = getattr(canonical[name].rst, method)(**options)
            output = output.rename(name).rename(dimensions).transpose(*source[name].dims)

            # Differentiate ops that transform coordinates or not (filter/fill_nodata don't, others do)
            if method in ("filter", "fill_nodata"):
                output = output.assign_coords(source[name].coords)
            elif coordinates == "cf":
                output.easting.attrs["axis"] = "X"
                output.northing.attrs["axis"] = "Y"
            outputs[name] = output

        # 3/ Compare values and metadata
        rtol = 0.0
        if lazy and method == "filter" and options["method"] == "mean":
            # Chunked means can add decimal slopes in a different order; allow four float64 rounding units
            rtol = float(4 * np.finfo(np.float64).eps)
        actual = result.compute() if lazy else result
        assert_dataset_output_equal(source, actual, outputs, rtol=rtol)
        if method == "sieve":
            # Check that the isolated cell merged into the background, using each date's upper-left label
            expected_labels = np.broadcast_to(canonical.regions.data[..., :1, :1], actual.regions.shape)
            np.testing.assert_array_equal(actual.regions.data, expected_labels)
        xr.testing.assert_identical(source, original)
        if lazy:
            assert all(source[name].chunks for name in source.rst.variables)
            assert source.point_z.chunks

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "method,options,representation,dimensions,variables",
        [
            ("interp_points", {"method": "nearest"}, "dataarray", "native", None),
            ("interp_at_points", {"method": "nearest"}, "dataset", "cf", None),
            ("interp_at_points", {"method": "nearest"}, "dataset_xy_variables", "native", None),
            ("interp_at_points", {"method": "nearest"}, "geodataframe", "native", ["dem"]),
            ("interp_at_points", {"method": "nearest"}, "tuple", "native", ["dem"]),
            ("interp_at_points", {"method": "nearest"}, "tuple", "cf", None),
            ("resample_at_points", {"method": "nearest"}, "dataarray", "cf", None),
            ("reduce_at_points", {"window": 3}, "dataarray", "time_transposed", ["dem"]),
        ],
    )
    def test_methods__point_outputs(
        self,
        mixed_dataset: xr.Dataset,
        method: str,
        options: dict[str, Any],
        dimensions: str,
        representation: str,
        lazy: bool,
        variables: list[str] | None,
    ) -> None:
        """
        Checks that Dataset methods producing point cloud outputs match DataArray calls per recognized raster variable,
        and do not affect independent points and scalar variables in the dataset.

        We test methods with different point inputs, native/CF spatial coordinates and time labels.
        """

        # 1/ Create a test mixed Xarray Dataset by modifying the mixed_dataset fixture
        # We optionally define dates/cf dimensions to check for consistency there
        source, canonical = mixed_dataset, mixed_dataset.dem
        options = options.copy()
        if dimensions == "cf":
            source = source.rename({"x": "easting", "y": "northing"})
            source.easting.attrs["axis"] = "X"
            source.northing.attrs["axis"] = "Y"
        elif dimensions == "time_transposed":
            dates = xr.IndexVariable("time", np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"))
            slices = [canonical, canonical.copy(data=canonical.data + 100)]
            dem = xr.concat(slices, dim=dates).transpose("x", "time", "y")
            source = source.assign(dem=dem, temperature=("time", [12.0, 14.0]))
            canonical = dem.rename(time="band").transpose("band", "y", "x")
            options["band"] = 2
        if lazy:
            pytest.importorskip("dask")
            from dask.callbacks import Callback

            # Chunk both raster and point arrays, then prepare the requested point input representation
            chunks = {"northing": 3, "easting": 4, "point": 4} if dimensions == "cf" else {"y": 3, "x": 4, "point": 4}
            source = source.chunk(chunks)
            if representation == "geodataframe":
                # Supply loaded X/Y so GeoDataFrame coordinates can be compared without reading Dask arrays
                source = source.assign_coords(x_point=mixed_dataset.x_point, y_point=mixed_dataset.y_point)
        original = source.copy(deep=True)

        # 2/ Test point inputs using all representations (DataArray, Dataset, GeoDataFrame or tuple of X/Y arrays)
        if representation in ("dataset", "dataset_xy_variables"):
            points = source[["point_z", "point_sigma"]]
            if representation == "dataset_xy_variables":
                points = points.reset_coords(["x_point", "y_point"])
        elif representation == "geodataframe":
            points = mixed_dataset.point_z.pc.to_geoutils().gdf
        elif representation == "tuple":
            points = (source.x_point.data, source.y_point.data)
        else:
            points = source.point_z

        # 3/ Run Dataset and DataArray methods on the same points and band, then compare exact output equality
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.rst, method)(points, variables=variables, **options)
        names = source.rst.variables if variables is None else variables
        if lazy:
            # Check that sampling runs no tasks and leaves the point results and original point graph lazy
            assert not tasks
            for name in names:
                assert source[name].chunks and result[name].chunks
                assert result[name].dims == ("point",)
            assert result.point_z.data is source.point_z.data
            for name in source.rst.variables:
                if name not in names:
                    assert result[name].data is source[name].data
        reference_points = points.pc.to_xarray() if representation == "geodataframe" else mixed_dataset.point_z
        reference_method = "interp_at_points" if method == "interp_points" else method
        outputs = {}
        for name in names:
            reference_raster = canonical if name == "dem" else mixed_dataset[name]
            expected = getattr(reference_raster.rst, reference_method)(reference_points, **options)
            outputs[name] = expected.rename({"x": "x_point", "y": "y_point"}).rename(name)
        assert_dataset_output_equal(source, result.compute() if lazy else result, outputs)
        xr.testing.assert_identical(source, original)
        if lazy:
            assert all(source[name].chunks for name in source.rst.variables)
            assert source.point_z.chunks
            if representation != "geodataframe":
                assert source.x_point.chunks and source.y_point.chunks


    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "method,options,with_mask,compare_outputs",
        [
            pytest.param(
                "variogram",
                {
                    "band": 1,
                    "n_pairs": 12,
                    "strategy": "independent",
                    "bins": [0, 2, 4, 8],
                    "min_lag": 0.1,
                    "max_lag": 8,
                    "estimator": np.mean,
                    "n_runs": 2,
                    "random_state": 7,
                },
                True,
                assert_variograms_equal,
                id="variogram",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "variables,dimensions,random_state",
        [
            (None, "native", "seed"),
            (["slope", "dem"], "transposed", "generator"),
            (["slope"], "cf", "seed"),
            (None, "time", "generator"),
        ],
    )
    def test_methods__dict_outputs(
        self,
        mixed_dataset: xr.Dataset,
        method: str,
        options: dict[str, Any],
        with_mask: bool,
        compare_outputs: Callable[[dict[str, Any], dict[str, Any]], None],
        variables: list[str] | None,
        dimensions: str,
        random_state: str,
        lazy: bool,
    ) -> None:
        """
        Checks that Dataset methods returning dictionaries match separate DataArray calls for the selected rasters.

        We test variable selection, native/CF raster dimensions and time labels, with shared masks, bins and random
        states when supplied.
        """

        # 1/ Create a test mixed Xarray Dataset by modifying the mixed_dataset fixture
        # We copy the mixed dataset and add NaNs at different pixels
        source = mixed_dataset.copy(deep=True)
        source.dem.data[1, 2] = np.nan
        source.slope.data[3, 4] = np.nan
        canonical = source[["dem", "slope"]]
        band = 1

        # Prepare date, transposed or CF dimensions while preserving a band/y/x reference
        if dimensions == "time":
            canonical = canonical.expand_dims(band=[1, 2]).copy(deep=True)
            for name in canonical.data_vars:
                canonical[name].data[1] *= 2
            dates = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]")
            rasters = canonical.rename(band="time").assign_coords(time=dates)
            source = source.assign({name: rasters[name] for name in rasters.data_vars})
            band = 2
        elif dimensions == "transposed":
            source = source.transpose("x", "y", "point", missing_dims="ignore")
        elif dimensions == "cf":
            source = source.rename({"x": "easting", "y": "northing"})
            source.easting.attrs["axis"] = "X"
            source.northing.attrs["axis"] = "Y"
        if lazy:
            from unittest.mock import Mock

            pytest.importorskip("dask")
            from dask import array as dask_array
            from dask import delayed
            from dask.callbacks import Callback

            # Use shorter edge chunks and add a raster that raises an error if it is read
            names = source.rst.variables if variables is None else variables
            chunks = {"northing": 3, "easting": 4, "point": 4} if dimensions == "cf" else {"y": 3, "x": 4, "point": 4}
            source = source.chunk(chunks)
            read_unused = Mock(side_effect=AssertionError("Variograms must not read unselected rasters."))
            unused = dask_array.from_delayed(delayed(read_unused)(), shape=source.dem.shape, dtype=source.dem.dtype)
            source = source.assign(unused=source.dem.copy(data=unused))
            variables = names
        original = source.copy(deep=True)

        # Copy the method options and select the requested time slice when the method accepts a band
        options = options.copy()
        if "band" in options:
            options["band"] = band
        reference_options = options.copy()

        # Create matching masks on native and canonical grids when the method case requests one
        if with_mask:
            mask = canonical.dem.isel(band=0, drop=True) > 15 if dimensions == "time" else canonical.dem > 15
            source_mask = source.dem.isel(time=0, drop=True) > 15 if dimensions == "time" else source.dem > 15
            reference_options["mask"] = mask
            options["mask"] = source_mask

        # 2/ Run Dataset and DataArray methods with the same mask, bins and random state
        # Run the DataArray method per variable name
        names = source.rst.variables if variables is None else variables
        if "random_state" in reference_options:
            reference_options["random_state"] = 7 if random_state == "seed" else np.random.default_rng(7)
        expected = {name: getattr(canonical[name].rst, method)(**reference_options) for name in names}

        # We reset the random state and pass bin edges as an iterator to test reuse across variables and runs
        if "random_state" in options:
            options["random_state"] = 7 if random_state == "seed" else np.random.default_rng(7)
        if "bins" in options and not isinstance(options["bins"], str):
            options["bins"] = iter(options["bins"])

        # Run the Dataset method while recording any reads of chunked inputs
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.rst, method)(variables=variables, **options)
        if lazy:
            # Check that selected tasks ran without reading the unused raster or changing the lazy source arrays
            assert tasks
            assert not read_unused.called
            assert all(source[name].chunks for name in source.rst.variables)
            assert source.point_z.chunks
            if method == "variogram":
                assert all(isinstance(variogram.semivariance, np.ndarray) for variogram in result.values())

        # 3/ Check exact equality
        assert isinstance(result, dict)
        assert list(result) == names
        compare_outputs(result, expected)
        xr.testing.assert_identical(source, original)

    @pytest.mark.parametrize("as_keyword", [False, True])
    @pytest.mark.parametrize("dimensions", ["native", "cf_transposed"])
    def test_reproject__native_reference(self, mixed_dataset: xr.Dataset, dimensions: str, as_keyword: bool) -> None:
        """
        Checks that Dataset reprojection accepts a native raster DataArray as its match-reference argument.

        We need this additional test for using a raster variable from a Dataset as the reference during reprojection.
        """

        # We create a coarser reference
        reference = mixed_dataset.dem.rst.reproject(res=2)
        native = reference
        if dimensions == "cf_transposed":
            native = reference.rename({"x": "easting", "y": "northing"}).transpose("easting", "northing")
            native.easting.attrs["axis"] = "X"
            native.northing.attrs["axis"] = "Y"
        original = native.copy(deep=True)

        # We reproject using this reference, with/without keyword to check positional argument too
        result = mixed_dataset.rst.reproject(ref=native) if as_keyword else mixed_dataset.rst.reproject(native)

        # Check exact equality
        assert result.dem.rst.transform == reference.rst.transform
        assert result.dem.rst.crs == reference.rst.crs
        xr.testing.assert_identical(result.dem, mixed_dataset.dem.rst.reproject(ref=reference).rename("dem"))
        xr.testing.assert_identical(native, original)

    @pytest.mark.parametrize("variables", [None, ["dem", "slope"]])
    def test_reproject__independent_grids(self, mixed_dataset: xr.Dataset, variables: list[str] | None) -> None:
        """
        Checks that multiple raster grids can be transformed independently.
        """

        # We create a coarse grid, merge it with the mixed dataset, to have 2 independent raster grids in the dataset
        coarse = gu.DataArrayRasterAccessor.from_array(np.arange(12.0).reshape(3, 4), from_origin(0, 6, 2, 2), 32632)
        coarse = coarse.rio.write_crs(32632, grid_mapping_name="coarse_ref").drop_vars("spatial_ref")
        coarse = coarse.rename({"x": "coarse_x", "y": "coarse_y"})
        coarse.coarse_x.attrs["axis"] = "X"
        coarse.coarse_y.attrs["axis"] = "Y"
        source = xr.merge([mixed_dataset, coarse.to_dataset(name="coarse")], compat="no_conflicts")
        original = source.copy(deep=True)

        # We reproject through the dataset
        result = source.rst.reproject(crs=4326, nodata=-99999, variables=variables)

        # Then we reproject through every data array, and compare for exact equality
        for name in source.rst.variables:
            if variables is None or name in variables:
                raster = source[name]
                dimensions = {raster.rio.x_dim: "x", raster.rio.y_dim: "y"}
                expected = raster.rename(dimensions).rst.reproject(crs=4326, nodata=-99999)
                np.testing.assert_array_equal(result[name].data, expected.data)
                assert result[name].rio.transform() == expected.rst.transform
                assert result[name].rio.crs.to_epsg() == 4326
            else:
                xr.testing.assert_identical(result[name].variable, source[name].variable)
                xr.testing.assert_identical(result.coarse_ref.variable, source.coarse_ref.variable)
                assert result[name].rio.crs.to_epsg() == 32632
            assert result[name].encoding["grid_mapping"] == source[name].encoding["grid_mapping"]
        xr.testing.assert_identical(result.point_z.variable, source.point_z.variable)
        xr.testing.assert_identical(source, original)

    @pytest.mark.parametrize(
        "method,options,positional",
        [("fill_nodata", {"interpolation": "nearest"}, False),
         ("clip", {}, False),
         ("clip", {}, True)],
    )
    def test_methods__mask(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], positional: bool
    ) -> None:
        """Checks that all Dataset operations requiring a mask work even when renaming/reordering dimensions."""

        # We create a mixed dataset with a NaN for interpolation
        mixed_dataset.dem.data[2, 3] = np.nan

        # Now let's rename and transpose the raster axes
        source = mixed_dataset.rename({"x": "easting", "y": "northing"})
        source.easting.attrs["axis"] = "X"
        source.northing.attrs["axis"] = "Y"
        source = source.transpose("easting", "northing", "point", missing_dims="ignore")

        # We create a mask, and use it in the operation being tested
        original = source.copy(deep=True)
        mask, canonical_mask = source.dem > 15, mixed_dataset.dem > 15
        if positional:
            result = getattr(source.rst, method)(mask, **options)
        else:
            result = getattr(source.rst, method)(mask=mask, **options)

        # Finally, we check exact equality with the DataArray operation
        for name in source.rst.variables:
            expected = getattr(mixed_dataset[name].rst, method)(mask=canonical_mask, **options)
            np.testing.assert_array_equal(result[name].transpose("northing", "easting").data, expected.data)
            assert result[name].rio.transform() == expected.rst.transform

        # And check that point values and source dataset are unchanged
        xr.testing.assert_identical(result.point_z.variable, source.point_z.variable)
        xr.testing.assert_identical(source, original)


    @pytest.mark.parametrize("dimensions", ["native", "transposed", "cf"])
    def test_stats__shared_grouping(self, mixed_dataset: xr.Dataset, dimensions: str) -> None:
        """
        Checks that Dataset statistics on rasters share grouping calculation.

        We need this test for the Dataset logic that passes several rasters and their shared groups to one stats() call.
        """

        # We prepare the mixed dataset with native/transposed/CF raster dimensions
        source = mixed_dataset
        if dimensions == "transposed":
            source = source.transpose("x", "y", "point", missing_dims="ignore")
        elif dimensions == "cf":
            source = source.rename({"x": "easting", "y": "northing"})
            source.easting.attrs["axis"] = "X"
            source.northing.attrs["axis"] = "Y"
        original = source.copy(deep=True)

        # Then group both rasters by DEM values, and compute stats
        groups = source.dem > 15
        options = {"by": {"high": groups}, "categories": {"high": [False, True]}}
        result = source.rst.stats("mean", **options)

        # We run the DataArray calculation, passing values from variables all at once
        # (should be equivalent of Dataset behaviour)
        options["by"] = {"high": mixed_dataset.dem > 15}
        expected = mixed_dataset.dem.rst.stats(
            "mean", values={"dem": mixed_dataset.dem, "slope": mixed_dataset.slope}, **options
        )
        import pandas as pd

        # Check exact statistics equality, and check unchanged source dataset
        pd.testing.assert_frame_equal(result, expected, check_exact=True)
        xr.testing.assert_identical(source, original)

    def test_stats__mask_and_reference(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks that Dataset statistics normalize a CF mask and reference grid before selecting raster values.

        We need to test this separately because stats() on a Dataset aligns inputs separately for efficiency.
        """

        # We rename and transpose the fixture raster axes to test CF inputs with a different dimension order
        source = mixed_dataset.rename({"x": "easting", "y": "northing"}).transpose(
            "easting", "northing", "point", missing_dims="ignore"
        )
        source.easting.attrs["axis"] = "X"
        source.northing.attrs["axis"] = "Y"
        original = source.copy(deep=True)

        # Create a coarser reference and put the mask on its grid to isolate Dataset input normalization
        reference = mixed_dataset.dem.rst.reproject(res=2)
        native_reference = reference.rename({"x": "easting", "y": "northing"}).transpose("easting", "northing")
        native_reference.easting.attrs["axis"] = "X"
        native_reference.northing.attrs["axis"] = "Y"
        mask = reference > 15
        native_mask = native_reference > 15
        names = ["slope", "dem"]
        options: dict[str, Any] = {"align": "reproject", "interpolation": "nearest"}

        # Calculate means through both Dataset and DataArray
        result = source.rst.stats("mean", variables=names, at=native_reference, mask=native_mask, **options)
        values = {name: mixed_dataset[name] for name in names}
        expected = mixed_dataset.dem.rst.stats("mean", values=values, at=reference, mask=mask, **options)

        # Check exact means, selected variable order and the unchanged source dataset
        assert result == expected
        assert list(result) == names
        xr.testing.assert_identical(source, original)

    def test_stats__fractional_coverage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that stats(by=vector) on a Dataset computes fractional polygon coverage only once."""

        import geopandas as gpd
        from rasterio.transform import from_origin
        from shapely.geometry import box

        import geoutils.stats.fractional as fractional

        # We create two rasters and a polygon with fractional coverage
        raster = gu.DataArrayRasterAccessor.from_array(
            np.array([[1.0, 2.0], [3.0, 4.0]]), from_origin(0, 2, 1, 1), 32631
        )
        source = xr.Dataset({"dem": raster, "double": raster.copy(data=2 * raster.data)})
        zones = gpd.GeoDataFrame({"zone": [0]}, geometry=[box(0.5, 0.5, 1.5, 1.5)], crs=32631)

        # We count calls to the polygon intersection internal function to check that
        # the Dataset method computes it only once
        intersect = fractional._grid_intersection_fractions
        calls = []
        def count_intersections(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return intersect(*args, **kwargs)

        # Then we run fractional statistics (with the intersection counter installed)
        monkeypatch.setattr(fractional, "_grid_intersection_fractions", count_intersections)
        result = source.rst.stats("mean", by={"zone": (zones, "zone")}, fractional=True, overlap_backend="shapely")

        # Check only one fractional coverage calculation ran
        assert len(calls) == 1

        # Compare with NumPy means (polygon covers each cell equally)
        np.testing.assert_array_equal(result[("dem", "mean")], [np.mean(source.dem.data)])
        np.testing.assert_array_equal(result[("double", "mean")], [np.mean(source.double.data)])

    @pytest.mark.parametrize("representation", ["dataarray", "dataset"])
    @pytest.mark.parametrize("scheduler", ["synchronous", "threads", "processes"])
    @pytest.mark.parametrize("compute", [True, False])
    def test_to_file__chunked_writing(
        self,
        representation: str,
        scheduler: str,
        compute: bool,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that chunk writing match eager files, and stays lazy."""

        dask = import_optional("dask")
        import dask.array as da
        from dask.callbacks import Callback

        # 1/ Create eager and chunked rasters
        # We create 2 bands with a NaN (to test nodata encoding when writing chunks)
        values = np.arange(70, dtype=np.float32).reshape(2, 5, 7) / 4
        values[0, 0, 0] = np.nan
        eager: xr.DataArray | xr.Dataset = gu.DataArrayRasterAccessor.from_array(
            values, from_origin(7, 46, 0.25, 0.25), 4326, nodata=-9999
        )
        if representation == "dataset":
            eager = xr.Dataset({"first": eager.isel(band=0, drop=True), "second": eager.isel(band=1, drop=True)})

        # Split the grid into chunks of 2/3 pixels (to leave shorter edge chunks, to check for incorrect writes)
        source = eager.chunk({"y": 2, "x": 3})
        if representation == "dataset":
            source["second"] = source.second.chunk({"y": 3, "x": 2})

        # Check every source is lazy
        variables = source.data_vars.values() if isinstance(source, xr.Dataset) else (source,)
        original_chunks = [variable.chunks for variable in variables]
        assert all(not variable.rst.is_loaded for variable in variables)

        # Write eager reference file
        expected_file = tmp_path / "eager.tif"
        eager.rst.to_file(expected_file)
        output_file = tmp_path / "chunked.tif"

        # 2/ Write chunked rasters with the selected scheduler
        task_keys = []
        def fail_array_conversion(*args: Any, **kwargs: Any) -> Any:
            """Fail if the writer gathers the Dask raster into a NumPy array."""
            raise AssertionError("to_file() must write chunks without converting the complete Dask array to NumPy")

        # We write with each scheduler
        # (Allows thread/process schedulers to test the shared file lock with two workers)
        with dask.config.set(scheduler=scheduler, num_workers=2), monkeypatch.context() as patch:
            patch.setattr(da.Array, "__array__", fail_array_conversion)
            with Callback(pretask=lambda key, graph, state: task_keys.append(key)):
                pending = source.rst.to_file(output_file, compute=compute)
                if compute:
                    # Check that a direct write executes tasks and returns no pending result
                    assert pending is None
                    assert task_keys
                else:
                    # Check that a deferred write stays lazy until we explicitly compute it
                    assert dask.is_dask_collection(pending)
                    assert not task_keys
                    assert all(not variable.rst.is_loaded for variable in variables)
                    pending.compute()
                    assert task_keys

        # 3/ Compare loading behaviour and written files
        # Check that writing leaves the source arrays lazy with their original chunks
        assert [variable.chunks for variable in variables] == original_chunks
        assert all(not variable.rst.is_loaded for variable in variables)

        # We can then open and check for exact equality (using Rasterio directly to avoid any decoding)
        with rio.open(output_file) as actual, rio.open(expected_file) as expected:
            np.testing.assert_array_equal(actual.read(), expected.read())
            assert actual.transform == expected.transform
            assert actual.crs == expected.crs
            assert actual.nodata == expected.nodata
            assert actual.count == expected.count

    @pytest.mark.parametrize(
        "method,options,variables",
        [
            ("filter", {"method": "mean", "size": 3}, ["dem"]),
            ("fill_nodata", {"interpolation": "nearest"}, ["dem"]),
            ("reproject", {"res": 2}, None),
        ],
    )
    def test_chunked_methods__netcdf_metadata(
        self,
        mixed_dataset: xr.Dataset,
        tmp_path: Path,
        method: str,
        options: dict[str, Any],
        variables: list[str] | None,
    ) -> None:
        """
        Checks that raster operations on a packed CF NetCDF file stay lazy and preserve the source metadata.

        We need this test to cover NetCDF encoding that is absent from the method combinations.
        """

        import_optional("dask")
        from dask.callbacks import Callback

        # 1/ Write and open a packed CF NetCDF file
        # We prepare the fixture's rasters with CF axes, array attributes and a missing DEM pixel
        source = mixed_dataset[["dem", "slope"]].rename({"x": "easting", "y": "northing"})
        source.easting.attrs = {"axis": "X", "valid_range": np.array([0.0, 7.0])}
        source.northing.attrs["axis"] = "Y"
        source.spatial_ref.attrs.pop("crs_wkt")
        source.spatial_ref.attrs.pop("spatial_ref")
        source.dem.data[2, 3] = np.nan

        # Write a packed integer NetCDF file that decodes to floating values and a NaN at the gap
        source.dem.attrs.pop("_FillValue")
        source.dem.encoding.update(dtype="int16", scale_factor=0.1, add_offset=100.0, _FillValue=-32768)
        filename = tmp_path / "survey.nc"
        source.to_netcdf(filename, engine="scipy")

        # Open the file eagerly and with shorter edge chunks to check operations at chunk boundaries
        open_options = {"engine": "scipy", "decode_coords": "all"}
        eager = xr.open_dataset(filename, **open_options).load()
        lazy = xr.open_dataset(filename, chunks={"northing": 3, "easting": 4}, **open_options)
        original = lazy.copy(deep=True)
        encodings = {name: variable.encoding.copy() for name, variable in lazy.variables.items()}

        # 2/ Run raster operations without reading the file
        # Select both rasters for reprojection because they share coordinates, otherwise process only the DEM
        names = lazy.rst.variables if variables is None else variables
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            result = getattr(lazy.rst, method)(variables=variables, **options)

        # Check lazy raster inputs/outputs, no executed tasks and unchanged graphs for unselected rasters
        assert not tasks
        assert all(lazy[name].chunks and result[name].chunks for name in names)
        for name in lazy.rst.variables:
            if name not in names:
                assert result[name].data is lazy[name].data

        # 3/ Compare eager outputs and source metadata
        # Compute and compare with the same eager operation, then check the source data, encoding and laziness
        expected = getattr(eager.rst, method)(variables=variables, **options)
        xr.testing.assert_identical(result.compute(), expected)
        xr.testing.assert_identical(lazy, original)
        assert all(lazy[name].encoding == encoding for name, encoding in encodings.items())
        assert lazy.dem.chunks and lazy.slope.chunks

        # Close both file-backed datasets
        eager.close()
        lazy.close()

    @pytest.mark.parametrize("raster_name", ["exploradores_aster_dem", "everest_landsat_rgb"])
    def test_accessors__file_backed_workflow(self, raster_name: str, tmp_path: Path) -> None:
        """Checks that real GeoTIFF/LAZ data stays lazy through both accessors."""

        import_optional("dask")
        import_optional("pyarrow")
        import_optional("laspy")
        from dask.callbacks import Callback

        # 1/ Open independent raster and point files
        # We merge raster and point data from different continents and CRSs to test independent georeferencing
        raster_path = examples.get_path_test(raster_name)
        point_path = examples.get_path_test("coromandel_lidar")
        raster = gu.open_raster(raster_path).to_dataset(name="image")
        points = gu.open_pointcloud(point_path, columns="all", as_type="dataset")
        eager = xr.merge([raster, points], compat="no_conflicts")
        # Open both files in chunks of 64/51 pixels and 2001 points, leaving shorter final blocks
        lazy_raster = gu.open_raster(raster_path, chunks={"x": 64, "y": 51}).to_dataset(name="image")
        lazy_points = gu.open_pointcloud(point_path, columns="all", chunks=2001, as_type="dataset")
        lazy = xr.merge([lazy_raster, lazy_points], compat="no_conflicts")
        original = lazy.copy(deep=True)
        encoding = lazy.image.encoding.copy()

        # 2/ Perform raster and point transformations without reading either file
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            result = lazy.rst.reproject(crs=4326, nodata=-99999).rst.filter("mean", size=3)
            result = result.pc.subsample(17, random_state=7).pc.reproject(crs=4326)

        # Check that the transformations run no tasks and leave both inputs and outputs lazy
        assert not tasks
        assert lazy.image.chunks and lazy.Z.chunks
        assert result.image.chunks and result.Z.chunks
        assert result.sizes["point"] == 17

        # Run the same transformation chain eagerly for the reference result
        expected = eager.rst.reproject(crs=4326, nodata=-99999).rst.filter("mean", size=3)
        expected = expected.pc.subsample(17, random_state=7).pc.reproject(crs=4326)

        # 3/ Compare computation/outputs
        actual = result.compute()
        xr.testing.assert_identical(actual, expected)
        xr.testing.assert_identical(lazy, original)
        assert lazy.image.encoding == encoding
        assert actual.image.rst.crs.to_epsg() == actual.Z.pc.crs.to_epsg() == 4326
        assert np.isfinite(actual.image.data).any()

        # Write in chunks and reopen lazily
        filename = tmp_path / "selected.parquet"
        actual.pc.to_parquet(filename, chunks=7)
        reopened = gu.open_pointcloud(filename, columns="all", chunks=7, as_type="dataset")
        assert reopened.Z.chunks and reopened.sizes["point"] == 17

        # Compare exact equality
        stored = reopened.compute()
        for name in actual.pc.variables:
            np.testing.assert_array_equal(stored[name].data, actual[name].data)
            assert stored[name].dtype == actual[name].dtype
        np.testing.assert_array_equal(stored.x_point.data, actual.x_point.data)
        np.testing.assert_array_equal(stored.y_point.data, actual.y_point.data)
        assert stored.Z.pc.crs == actual.Z.pc.crs
        assert lazy.image.chunks and lazy.Z.chunks and reopened.Z.chunks


class TestDatasetRasterAccessorErrors:
    """Test module for errors/warnings for the raster Dataset accessor ``rst``."""

    @pytest.mark.parametrize(
        "method,options,message",
        [
            ("reproject", {"res": 2, "inplace": True}, "inplace=True"),
            ("filter", {"method": "mean", "size": 3, "as_array": True}, "as_array=True"),
            ("interp_at_points", {"as_array": True}, "as_array=True"),
            ("stats", {"values": ["dem"]}, "Use 'variables'"),
        ],
    )
    def test_methods__error_dataset_options(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], message: str
    ) -> None:
        """Checks an error is raised for invalid options specific to a Dataset accessor."""

        if method == "interp_at_points":
            options = dict(options, points=mixed_dataset.point_z)
        original = mixed_dataset.copy(deep=True)
        with pytest.raises(ValueError, match=message):
            getattr(mixed_dataset.rst, method)(**options)

        # Check unchanged source dataset after error
        xr.testing.assert_identical(mixed_dataset, original)

    @pytest.mark.parametrize(
        "method,options,variables,message",
        [
            ("reproject", {"res": 2}, [], "nonempty"),
            ("variogram", {}, ["missing"], "Unknown"),
            ("interp_points", {}, ["point_z"], "incompatible"),
            ("filter", {"method": "mean", "size": 3}, ["dem", "dem"], "distinct"),
        ],
    )
    def test_methods__error_variables(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], variables: list[str], message: str
    ) -> None:
        """Checks an error is raised for invalid variable selections."""

        if method == "interp_points":
            options = dict(options, points=mixed_dataset.point_z)
        with pytest.raises(ValueError, match=message):
            getattr(mixed_dataset.rst, method)(variables=variables, **options)

    @pytest.mark.parametrize("method,options", [("reproject", {"res": 2}), ("crop", {"bbox": (1, 1, 5, 4)})])
    def test_methods__error_shared_grid(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any]
    ) -> None:
        """Checks an error is raised when an unselected raster would be attached to new axes coordinates."""

        # Change only "dem" but not "slope" (that shares the same exes)
        with pytest.raises(ValueError, match="Untouched variables.*slope"):
            getattr(mixed_dataset.rst, method)(variables=["dem"], **options)

    @pytest.mark.parametrize("name", ["scan_line", "cell_bounds", "quality"])
    def test_reproject__error_auxiliary_axis(self, mixed_dataset: xr.Dataset, name: str) -> None:
        """Checks an error is raised for an auxiliary variable sharing partially a changed raster axis."""

        # We add either a row attribute, CF cell bounds or a categorical layer sharing the raster axis
        variables = {
            "scan_line": ("y", np.arange(5)),
            "cell_bounds": (("x", "vertex"), np.column_stack([np.arange(7), np.arange(7) + 1])),
            "quality": (("y", "x"), np.full((5, 7), "good")),
        }
        source = mixed_dataset.assign({name: variables[name]})

        # Reproject and check an error is raised because the unselected auxiliary variable uses the changed axis
        with pytest.raises(ValueError, match=f"Untouched variables.*{name}"):
            source.rst.reproject(res=2)

    def test_interp_at_points__error_lazy_geodataframe(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised with a lazy GeoDataFrame point input."""

        dgpd = import_optional("dask_geopandas")
        from dask.callbacks import Callback

        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor
        _register_dask_pointcloud_accessor()
        points = mixed_dataset.point_z.pc.to_geoutils().gdf
        lazy_points = dgpd.from_geopandas(points, npartitions=2)

        # Record tasks to check that raising the error does not load point partitions
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            with pytest.raises(ValueError, match="Convert lazy GeoDataFrame points explicitly"):
                mixed_dataset.rst.interp_at_points(lazy_points, variables=["dem"], method="nearest")

        # Check no tasks ran and the point input remains lazy
        assert not tasks
        assert not lazy_points.pc.is_loaded
