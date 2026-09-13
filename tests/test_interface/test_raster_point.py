"""Tests for conversion between rasters and regular point clouds."""

from __future__ import annotations

import re
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

import geoutils as gu
from geoutils import examples, open_raster
from geoutils._dispatch import is_dask_dataframe
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster


class TestRasterPoint:
    """
    Test module for the exact conversions point-raster in to_pointcloud() and from_pointcloud_regular().

    Support for Dask/Multiproc is tested in TestRasterPointChunked further below.
    """

    # Paths to example data
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_to_pointcloud(self) -> None:
        """Test to_pointcloud method."""

        # 1/ Single band synthetic data

        # Create a small raster to test point sampling on
        img_arr = np.arange(25, dtype="int32").reshape(5, 5)
        img0 = gu.Raster.from_array(img_arr, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)

        # Sample the whole raster (fraction==1)
        points = img0.to_pointcloud()
        points_arr = img0.to_pointcloud(as_array=True)

        # Check output types
        assert isinstance(points, gu.Vector)
        assert isinstance(points_arr, np.ndarray)

        # Check that both outputs (array or vector) are fully consistent, order matters here
        assert np.array_equal(points.ds.geometry.x.values, points_arr[:, 0])
        assert np.array_equal(points.ds.geometry.y.values, points_arr[:, 1])
        assert np.array_equal(points.ds["b1"].values, points_arr[:, 2])

        # Validate that 25 points were sampled (equating to img1.height * img1.width) with x, y, and band0 values.
        assert points_arr.shape == (25, 3)
        assert points.ds.shape == (25, 2)  # One less column here due to geometry storing X and Y
        # Check that X, Y and Z arrays are equal to raster array input independently of value order
        x_coords, y_coords = img0.ij2xy(i=np.arange(0, 5), j=np.arange(0, 5))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 0])), np.sort(np.tile(x_coords, 5)))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 1])), np.sort(np.tile(y_coords, 5)))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 2])), np.sort(img_arr.ravel()))

        # Check that subsampling works properly
        points_arr = img0.to_pointcloud(subsample=0.2, as_array=True)
        assert points_arr.shape == (5, 3)

        # All values should be between 0 and 25
        assert all(0 <= points_arr[:, 2]) and all(points_arr[:, 2] < 25)

        # 2/ Multi-band synthetic data
        img_arr = np.arange(25, dtype="int32").reshape(5, 5)
        img_3d_arr = np.stack((img_arr, 25 + img_arr, 50 + img_arr), axis=0)
        img3d = gu.Raster.from_array(img_3d_arr, transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326)

        # Sample the whole raster (fraction==1)
        points = img3d.to_pointcloud(auxiliary_data_bands=[2, 3])
        points_arr = img3d.to_pointcloud(as_array=True, auxiliary_data_bands=[2, 3])

        # Check equality between both output types
        assert np.array_equal(points.ds.geometry.x.values, points_arr[:, 0])
        assert np.array_equal(points.ds.geometry.y.values, points_arr[:, 1])
        assert np.array_equal(points.ds["b1"].values, points_arr[:, 2])
        assert np.array_equal(points.ds["b2"].values, points_arr[:, 3])
        assert np.array_equal(points.ds["b3"].values, points_arr[:, 4])

        # Check it is the right data
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 0])), np.sort(np.tile(x_coords, 5)))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 1])), np.sort(np.tile(y_coords, 5)))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 2])), np.sort(img_3d_arr[0, :, :].ravel()))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 3])), np.sort(img_3d_arr[1, :, :].ravel()))
        assert np.array_equal(np.sort(np.asarray(points_arr[:, 4])), np.sort(img_3d_arr[2, :, :].ravel()))

        # With a subsample
        points_arr = img3d.to_pointcloud(as_array=True, subsample=10, auxiliary_data_bands=[2, 3])
        assert points_arr.shape == (10, 5)

        # Check the values are still good
        assert all(0 <= points_arr[:, 2]) and all(points_arr[:, 2] < 25)
        assert all(25 <= points_arr[:, 3]) and all(points_arr[:, 3] < 50)
        assert all(50 <= points_arr[:, 4]) and all(points_arr[:, 4] < 75)

        # 3/ Single-band real raster with nodata values
        img1 = gu.Raster(self.aster_dem_path)

        # Get a large sample to ensure they should be some NaNs normally
        points_arr = img1.to_pointcloud(subsample=10000, as_array=True, random_state=42)
        points = img1.to_pointcloud(subsample=10000, random_state=42)

        # This should not load the image
        assert not img1.is_loaded

        # The subsampled values should be valid and the right shape
        assert points_arr.shape == (10000, 3)
        assert points.ds.shape == (10000, 2)  # One less column here due to geometry storing X and Y
        assert all(np.isfinite(points_arr[:, 2]))

        # The output should respect the default band naming and the input CRS
        assert np.array_equal(points.ds.columns, ["b1", "geometry"])
        assert points.crs == img1.crs

        # Try setting the band name
        points = img1.to_pointcloud(data_column_name="lol", subsample=10)
        assert np.array_equal(points.ds.columns, ["lol", "geometry"])

        # Keeping the nodata values
        points_invalid = img1.to_pointcloud(subsample=10000, random_state=42, skip_nodata=False)

        # The subsampled values should not all be valid and the right shape
        assert points_invalid.ds.shape == (10000, 2)  # One less column here due to geometry storing X and Y
        assert any(~np.isfinite(points_invalid["b1"].values))

        # 4/ Multi-band real raster
        img2 = gu.Raster(self.landsat_rgb_path)

        # By default only loads a single band without loading
        points_arr = img2.to_pointcloud(subsample=10, as_array=True)
        points = img2.to_pointcloud(subsample=10)

        assert points_arr.shape == (10, 3)
        assert points.ds.shape == (10, 2)  # One less column here due to geometry storing X and Y
        assert not img2.is_loaded

        # Storing auxiliary bands
        points_arr = img2.to_pointcloud(subsample=10, as_array=True, auxiliary_data_bands=[2, 3])
        points = img2.to_pointcloud(subsample=10, auxiliary_data_bands=[2, 3])
        assert points_arr.shape == (10, 5)
        assert points.ds.shape == (10, 4)  # One less column here due to geometry storing X and Y
        assert not img2.is_loaded
        assert np.array_equal(points.ds.columns, ["b1", "b2", "b3", "geometry"])

        # Try setting the column name of a specific band while storing all
        points = img2.to_pointcloud(subsample=10, data_column_name="yes", data_band=2, auxiliary_data_bands=[1, 3])
        assert np.array_equal(points.ds.columns, ["yes", "b1", "b3", "geometry"])

        # 5/ Error raising
        with pytest.raises(ValueError, match="Data column name must be a string.*"):
            img1.to_pointcloud(data_column_name=1)  # type: ignore
        with pytest.raises(
            ValueError,
            match=re.escape("Data band number must be an integer between 1 and the total number of bands (3)."),
        ):
            img2.to_pointcloud(data_band=4)
        with pytest.raises(
            ValueError, match="Passing auxiliary column names requires passing auxiliary data band numbers as well."
        ):
            img2.to_pointcloud(auxiliary_column_names=["a"])
        with pytest.raises(
            ValueError, match="Auxiliary data band number must be an iterable containing only integers."
        ):
            img2.to_pointcloud(auxiliary_data_bands=[1, 2.5])  # type: ignore
            img2.to_pointcloud(auxiliary_data_bands="lol")  # type: ignore
        with pytest.raises(
            ValueError,
            match=re.escape("Auxiliary data band numbers must be between 1 and the total number of bands (3)."),
        ):
            img2.to_pointcloud(auxiliary_data_bands=[0])
            img2.to_pointcloud(auxiliary_data_bands=[4])
        with pytest.raises(
            ValueError, match=re.escape("Main data band 1 should not be listed in auxiliary data bands [1, 2].")
        ):
            img2.to_pointcloud(auxiliary_data_bands=[1, 2])
        with pytest.raises(ValueError, match="Auxiliary column names must be an iterable containing only strings."):
            img2.to_pointcloud(auxiliary_data_bands=[2, 3], auxiliary_column_names=["lol", 1])
        with pytest.raises(
            ValueError, match="Length of auxiliary column name and data band numbers should be the same*"
        ):
            img2.to_pointcloud(auxiliary_data_bands=[2, 3], auxiliary_column_names=["lol", "lol2", "lol3"])

    def test_from_pointcloud(self) -> None:
        """Test from_pointcloud method."""

        # 1/ Create a small raster to test point sampling on
        shape = (5, 5)
        nodata = 100
        img_arr = np.arange(np.prod(shape), dtype="int32").reshape(shape)
        transform = rio.transform.from_origin(0, 5, 1, 1)
        img1 = gu.Raster.from_array(img_arr, transform=transform, crs=4326, nodata=nodata)

        # Check both inputs work (grid coords or transform+shape) on a subsample
        pc1 = img1.to_pointcloud(subsample=10)
        img1_sub = gu.Raster.from_pointcloud_regular(pc1, transform=transform, shape=shape)

        grid_coords1 = img1.coords(grid=False)
        img1_sub2 = gu.Raster.from_pointcloud_regular(pc1, grid_coords=grid_coords1)

        assert img1_sub.raster_equal(img1_sub2)

        # Check that number of valid values are equal to point cloud size
        assert np.count_nonzero(~img1_sub.data.mask) == 10

        # With no subsampling, should get the exact same raster back
        pc1_full = img1.to_pointcloud()
        img1_full = gu.Raster.from_pointcloud_regular(pc1_full, transform=transform, shape=shape, nodata=nodata)
        assert img1.raster_equal(img1_full, warn_failure_reason=True)

        # 2/ Single-band real raster with nodata values
        img2 = gu.Raster(self.aster_dem_path)
        nodata = img2.nodata
        transform = img2.transform
        shape = img2.shape

        # Check both inputs work (grid coords or transform+shape) on a subsample
        pc2 = img2.to_pointcloud(subsample=10000, random_state=42)
        img2_sub = gu.Raster.from_pointcloud_regular(pc2, transform=transform, shape=shape, nodata=nodata)

        grid_coords2 = img2.coords(grid=False)
        img2_sub2 = gu.Raster.from_pointcloud_regular(pc2, grid_coords=grid_coords2, nodata=nodata)

        assert img2_sub.raster_equal(img2_sub2, warn_failure_reason=True)

        # Check that number of valid values are equal to point cloud size
        assert np.count_nonzero(~img2_sub.data.mask) == 10000

        # With no subsampling, should get the exact same raster back
        pc2_full = img2.to_pointcloud()
        img2_full = gu.Raster.from_pointcloud_regular(pc2_full, transform=transform, shape=shape, nodata=nodata)
        assert img2.raster_equal(img2_full, warn_failure_reason=True, strict_masked=False)

        # 3/ Error raising
        with pytest.raises(TypeError, match="Input grid coordinates must be 1D arrays.*"):
            gu.Raster.from_pointcloud_regular(pc1, grid_coords=(1, "lol"))  # type: ignore
        with pytest.raises(ValueError, match="Grid coordinates must be regular*"):
            grid_coords1[0][0] += 1
            gu.Raster.from_pointcloud_regular(pc1, grid_coords=grid_coords1)  # type: ignore
        with pytest.raises(
            ValueError, match="Either grid coordinates or both geotransform and shape must be provided."
        ):
            gu.Raster.from_pointcloud_regular(pc1)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestRasterPointChunked:
    """
    Test module for to_pointcloud() across eager, Dask and Multiprocessing backends.

    We check that Dask outputs stay lazy until explicitly computed, and Multiprocessing input/outputs stay unloaded,
    both returning the exact same result.
    """

    @pytest.mark.parametrize("subsample", [1, 11])
    @pytest.mark.parametrize("skip_nodata", [True, False])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_to_pointcloud__chunked_backends_equal(
        self, subsample: int, skip_nodata: bool, as_array: bool, tmp_path: Path
    ) -> None:
        """Checks that eager, Dask and Multiprocessing conversions return exactly the same values."""

        import dask.array as da

        # Write a raster file with three bands, one nodata value pixel and a dimension size that will
        # make the final chunk a different size (not a multiple of the chunksize)
        main_values = np.arange(63, dtype=np.float32).reshape((7, 9))
        main_values[2, 3] = np.nan
        values = np.stack((main_values, main_values + 100, main_values + 200))
        transform = rio.transform.from_origin(500000, 8600000, 20, 20)
        source_file = tmp_path / "point-source.tif"
        gu.Raster.from_array(values, transform, 32633, nodata=-9999).to_file(source_file)

        # Open the same file as a loaded NumPy raster, a lazy Dask array and an unloaded Raster for multiprocessing
        eager = gu.Raster(source_file)
        eager.load()
        dask_source = open_raster(str(source_file), chunks={"band": 1, "x": 4, "y": 3})
        multiprocessing_source = gu.Raster(source_file)
        dask_source_array = dask_source.data
        options = {
            "subsample": subsample,
            "skip_nodata": skip_nodata,
            "as_array": as_array,
            "random_state": 42,
            "auxiliary_data_bands": [2, 3],
            "force_pixel_offset": "center",
        }

        # Run to_pointcloud() with every backend, matching MP chunks with Dask
        expected = eager.to_pointcloud(**options)
        dask_output = dask_source.rst.to_pointcloud(**options)
        point_output = tmp_path / f"points-{subsample}-{skip_nodata}.gpkg"
        with MpCluster({"nb_workers": 2}) as cluster:
            multiprocessing_output = multiprocessing_source.to_pointcloud(
                **options,
                mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(point_output), cluster=cluster),
            )

        # Dask keeps both source and output lazy until the result is explicitly computed
        assert dask_source.data is dask_source_array
        assert not dask_source._in_memory
        if as_array:
            assert isinstance(dask_output, da.Array)
            dask_computed = dask_output.compute()
            assert isinstance(dask_output, da.Array)
        else:
            assert is_dask_dataframe(dask_output)
            assert not dask_output.pc.is_loaded
            assert dask_output.pc.data_column == "b1"
            dask_computed = dask_output.compute()
            assert not dask_output.pc.is_loaded
        assert dask_source.data is dask_source_array
        assert not dask_source._in_memory

        # Multiprocessing keeps the source unloaded; array output is eager, while point output stays on disk
        assert not multiprocessing_source.is_loaded
        if as_array:
            assert isinstance(multiprocessing_output, np.ndarray)
            assert not point_output.exists()
            np.testing.assert_array_equal(expected, dask_computed)
            np.testing.assert_array_equal(expected, multiprocessing_output)
        else:
            assert not multiprocessing_output.is_loaded
            assert multiprocessing_output.name == str(point_output)
            assert point_output.exists()
            assert expected.vector_equal(dask_computed)
            assert expected.pointcloud_equal(multiprocessing_output)
            assert multiprocessing_output.is_loaded
            assert not multiprocessing_source.is_loaded

    def test_to_pointcloud__multiprocessing_empty_output_stays_unloaded(self, tmp_path: Path) -> None:
        """Checks that an all-nodata raster produces an empty file-backed PointCloud without loading its source."""

        # Write an all-nodata raster split into several uneven tiles
        source_file = tmp_path / "empty-source.tif"
        output_file = tmp_path / "empty-points.gpkg"
        values = np.ma.masked_all((5, 7), dtype=np.int16)
        values.data.fill(-9999)
        gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), 32633, nodata=-9999).to_file(source_file)
        source = gu.Raster(source_file)

        # Let each worker stage an empty point partition and assemble their common GeoPackage schema
        with MpCluster({"nb_workers": 2}) as cluster:
            result = source.to_pointcloud(
                mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file), cluster=cluster)
            )

        # Read the point count from file metadata, then load the empty table explicitly
        assert not source.is_loaded
        assert not result.is_loaded
        assert result.point_count == 0
        assert not result.is_loaded
        assert list(result.ds.columns) == ["b1", "geometry"]
        assert result.is_loaded

    def test_to_pointcloud__error_dask_with_multiproc(self) -> None:
        """Checks that trying to use Multiproc with a Dask raises an error, without loading the file."""

        import dask.array as da

        # Build a lazy Dask raster that will trigger Dask execution
        values = da.arange(20, chunks=7).reshape((4, 5))
        transform = rio.transform.from_origin(0, 4, 1, 1)
        source = gu.RasterAccessor.from_array(values, transform, 4326)
        source_array = source.data

        # Verify error is raised when passing mp_config, without loading or affecting the Dask object
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            source.rst.to_pointcloud(as_array=True, mp_config=MultiprocConfig(chunks=(2, 3)))
        assert source.data is source_array
        assert not source._in_memory
