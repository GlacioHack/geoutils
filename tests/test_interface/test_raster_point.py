"""Tests for conversion between rasters and regular point clouds."""

from __future__ import annotations

import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio as rio
from numpy.typing import NDArray
from shapely.geometry import Point

import geoutils as gu
from geoutils import examples, open_raster
from geoutils._dispatch import is_dask_dataframe
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster


class TestRasterPoint:
    """
    Test module for exact raster and regular point cloud conversions.

    These tests cover eager calls to to_pointcloud() and from_pointcloud_regular(). TestRasterPointChunked covers
    Dask and multiprocessing conversions, their loading behavior, and backend validation.
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

    def test_from_pointcloud_regular__error_invalid_grid_coordinates(self) -> None:
        """Checks that grid coordinates contain two regularly spaced increasing axes."""

        # Create a regular point cloud and its matching three by three coordinate vectors
        values = np.arange(9).reshape((3, 3))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 3, 1, 1), 32633)
        pointcloud = raster.to_pointcloud()
        x_coords, y_coords = raster.coords(grid=False)

        # Reject an irregular Y axis even when X remains regular
        irregular_y = y_coords.copy()
        irregular_y[1] += 0.25
        with pytest.raises(ValueError, match="Grid coordinates must be regular"):
            gu.Raster.from_pointcloud_regular(pointcloud, grid_coords=(x_coords, irregular_y))

        # Reject two irregular axes, axes without a measurable interval, and descending coordinates
        irregular_x = x_coords.copy()
        irregular_x[1] += 0.25
        with pytest.raises(ValueError, match="Grid coordinates must be regular"):
            gu.Raster.from_pointcloud_regular(pointcloud, grid_coords=(irregular_x, irregular_y))
        with pytest.raises(ValueError, match="at least two values"):
            gu.Raster.from_pointcloud_regular(pointcloud, grid_coords=(x_coords[:1], y_coords))
        with pytest.raises(ValueError, match="must increase"):
            gu.Raster.from_pointcloud_regular(pointcloud, grid_coords=(x_coords[::-1], y_coords))
        with pytest.raises(TypeError, match="must be 1D arrays"):
            gu.Raster.from_pointcloud_regular(pointcloud, grid_coords=(x_coords,))  # type: ignore[arg-type]

    def test_from_pointcloud_regular__error_point_outside_grid(self) -> None:
        """Checks that an aligned point outside the grid cannot wrap around an array edge."""

        # Create a complete two by two point cloud and move its first point one cell left of the grid
        transform = rio.transform.from_origin(0, 2, 1, 1)
        raster = gu.Raster.from_array(np.arange(4).reshape((2, 2)), transform, 32633)
        dataframe = raster.to_pointcloud().ds.copy()
        dataframe.loc[dataframe.index[0], "geometry"] = Point(-1, 2)

        # Reject the negative column instead of writing its value into the last raster column
        with pytest.raises(ValueError, match="fall outside the grid"):
            gu.Raster.from_pointcloud_regular(dataframe, transform=transform, shape=(2, 2))

    @pytest.mark.parametrize(
        "values",
        [
            np.array([[2**40 + 1, 2**40 + 3]], dtype=np.int64),
            np.array([[1.0 + 2**-30, 2.0 + 2**-29]], dtype=np.float64),
        ],
    )
    def test_to_pointcloud__skip_nodata_false_preserves_precision(self, values: NDArray[Any]) -> None:
        """Checks that keeping nodata cells does not reduce integer or floating-point precision."""

        # Convert values that float32 cannot represent exactly while retaining every raster cell
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 1, 1, 1), 32633)
        pointcloud = raster.to_pointcloud(skip_nodata=False)

        # The floating point output should represent every original value exactly
        assert pointcloud["b1"].dtype == np.dtype("float64")
        np.testing.assert_array_equal(pointcloud["b1"].to_numpy(), values.ravel().astype(np.float64))

    def test_to_pointcloud__iterable_auxiliary_columns(self) -> None:
        """Checks that iterable auxiliary bands and names are normalized before repeated use."""

        # Build three bands and pass their auxiliary options as one-use generators
        values = np.arange(12).reshape((3, 2, 2))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 32633)
        bands = (band for band in (2, 3))
        names = (name for name in ("second", "third"))
        pointcloud = raster.to_pointcloud(
            auxiliary_data_bands=bands,
            auxiliary_column_names=names,
        )

        # Preserve both auxiliary values and their requested column names
        assert list(pointcloud.ds.columns) == ["b1", "second", "third", "geometry"]
        np.testing.assert_array_equal(pointcloud["second"].to_numpy(), values[1].ravel())
        np.testing.assert_array_equal(pointcloud["third"].to_numpy(), values[2].ravel())

    @pytest.mark.parametrize(
        "options",
        [
            {"data_column_name": "b2", "auxiliary_data_bands": [2]},
            {"data_column_name": "geometry"},
        ],
    )
    def test_to_pointcloud__error_invalid_column_names(self, options: dict[str, Any]) -> None:
        """Checks that point data columns cannot be duplicated or replace the geometry column."""

        # Create two bands that would otherwise produce an ambiguous point dataframe
        values = np.arange(8).reshape((2, 2, 2))
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 32633)

        # Reject duplicate or reserved names before constructing the dataframe
        with pytest.raises(ValueError, match="must be unique"):
            raster.to_pointcloud(**options)


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
class TestRasterPointChunked:
    """
    Test module for to_pointcloud() across eager, Dask, and multiprocessing backends.

    Dask outputs must stay lazy until explicitly computed. Multiprocessing inputs and point outputs must stay unloaded,
    and both chunked backends must return the same values as an eager conversion.
    """

    @pytest.mark.parametrize("subsample", [1, 11, 17])
    @pytest.mark.parametrize("skip_nodata", [True, False])
    @pytest.mark.parametrize("as_array", [False, True])
    def test_to_pointcloud__chunked_backends_equal(
        self, subsample: int, skip_nodata: bool, as_array: bool, tmp_path: Path
    ) -> None:
        """Checks that eager, Dask and Multiprocessing conversions return the same point rows."""

        import dask.array as da

        # 1/ Write a three-band raster with one nodata pixel and shorter final row/column chunks
        main_values = np.arange(63, dtype=np.float32).reshape((7, 9))
        main_values[2, 3] = np.nan
        values = np.stack((main_values, main_values + 100, main_values + 200))
        transform = rio.transform.from_origin(500000, 8600000, 20, 20)
        source_file = tmp_path / "point-source.tif"
        gu.Raster.from_array(values, transform, 32633, nodata=-9999).to_file(source_file)

        # 2/ Open the file as a loaded raster, a lazy Dask array, and an unloaded multiprocessing source
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

        # 3/ Convert every backend with the same options and matching 3 x 4 chunks
        expected = eager.to_pointcloud(**options)
        dask_output = dask_source.rst.to_pointcloud(**options)
        point_output = tmp_path / f"points-{subsample}-{skip_nodata}.gpkg"
        with MpCluster({"nb_workers": 2}) as cluster:
            multiprocessing_output = multiprocessing_source.to_pointcloud(
                **options,
                mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(point_output), cluster=cluster),
            )

        # 4/ Check that the Dask source and output stay lazy until the result is computed explicitly
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

        # 5/ Check each computed result and the expected multiprocessing loading and file behavior
        assert not multiprocessing_source.is_loaded
        if as_array:
            assert isinstance(multiprocessing_output, np.ndarray)
            assert not point_output.exists()
            expected_order = np.lexsort((expected[:, 0], expected[:, 1]))
            dask_order = np.lexsort((dask_computed[:, 0], dask_computed[:, 1]))
            np.testing.assert_array_equal(expected[expected_order], dask_computed[dask_order])
            np.testing.assert_array_equal(expected, multiprocessing_output)
        else:
            assert not multiprocessing_output.is_loaded
            assert multiprocessing_output.name == str(point_output)
            assert point_output.exists()

            # Chunked outputs follow raster chunk order, so compare the same points after sorting by location
            expected_frame = expected.ds
            dask_frame = dask_computed
            multiprocessing_frame = multiprocessing_output.ds
            expected_order = np.lexsort((expected_frame.geometry.x, expected_frame.geometry.y))
            dask_order = np.lexsort((dask_frame.geometry.x, dask_frame.geometry.y))
            multiprocessing_order = np.lexsort((multiprocessing_frame.geometry.x, multiprocessing_frame.geometry.y))
            expected_frame = expected_frame.iloc[expected_order].reset_index(drop=True)
            dask_frame = dask_frame.iloc[dask_order].reset_index(drop=True)
            multiprocessing_frame = multiprocessing_frame.iloc[multiprocessing_order].reset_index(drop=True)
            np.testing.assert_array_equal(expected_frame.geometry.x, dask_frame.geometry.x)
            np.testing.assert_array_equal(expected_frame.geometry.y, dask_frame.geometry.y)
            np.testing.assert_array_equal(expected_frame.geometry.x, multiprocessing_frame.geometry.x)
            np.testing.assert_array_equal(expected_frame.geometry.y, multiprocessing_frame.geometry.y)
            for column in ("b1", "b2", "b3"):
                np.testing.assert_array_equal(expected_frame[column], dask_frame[column])
                np.testing.assert_array_equal(expected_frame[column], multiprocessing_frame[column])
            assert multiprocessing_output.is_loaded
            assert not multiprocessing_source.is_loaded

    def test_to_pointcloud__large_chunked_sample_preserves_precision(self, tmp_path: Path) -> None:
        """Checks that large Dask and multiprocessing samples preserve float64 raster values."""

        # Write distinct float64 values that would change if converted through float32
        values = np.arange(80, dtype=np.float64).reshape((8, 10)) + 2**-30
        source_file = tmp_path / "precise-point-source.tif"
        output_file = tmp_path / "precise-points.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)

        # Select more values than one 3 x 4 chunk through all three backends
        options = {"subsample": 17, "skip_nodata": False, "random_state": 42}
        expected = gu.Raster(source_file).to_pointcloud(**options)
        dask_result = open_raster(str(source_file), chunks={"x": 4, "y": 3}).rst.to_pointcloud(**options).compute()
        multiprocessing_source = gu.Raster(source_file)
        multiprocessing_result = multiprocessing_source.to_pointcloud(
            **options,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Keep exact source values and return the same selected values from every backend
        expected_values = expected["b1"].to_numpy()
        assert expected_values.dtype == np.dtype("float64")
        assert np.isin(expected_values, values).all()
        np.testing.assert_array_equal(np.sort(dask_result["b1"].to_numpy()), np.sort(expected_values))
        np.testing.assert_array_equal(np.sort(multiprocessing_result["b1"].to_numpy()), np.sort(expected_values))
        assert not multiprocessing_source.is_loaded

    def test_to_pointcloud__dask_large_sample_uses_cutoff(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a Dask sample larger than one raster chunk does not collect all selected cell indexes."""

        import dask.array as da

        # Write unique raster values and calculate the eager point rows selected by the same random keys
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "dask-cutoff-source.tif"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633)
        raster.to_file(source_file)
        expected = raster.to_pointcloud(subsample=17, random_state=42, as_array=True)

        # Reject the indexed path because 17 selected cells exceed each 3 x 4 input chunk
        from geoutils.interface import raster_point

        def reject_global_indexes(*args: Any, **kwargs: Any) -> Any:
            """Fail if Dask point conversion collects the complete sample indexes."""

            raise AssertionError("Large Dask point samples must use the bounded cutoff path.")

        monkeypatch.setattr(raster_point, "_sample_raster_cell_indices", reject_global_indexes)
        converted_chunks = []
        convert_chunk = raster_point._wrapper_raster_to_pointcloud_partition_dask

        def record_converted_chunk(*args: Any, **kwargs: Any) -> Any:
            """Record each point partition when Dask computes it."""

            converted_chunks.append(1)
            return convert_chunk(*args, **kwargs)

        monkeypatch.setattr(raster_point, "_wrapper_raster_to_pointcloud_partition_dask", record_converted_chunk)

        # Build and compute unknown-length point partitions without replacing or loading the source Dask array
        source = open_raster(str(source_file), chunks={"x": 4, "y": 3})
        source_data = source.data
        result = source.rst.to_pointcloud(subsample=17, random_state=42, as_array=True)
        assert isinstance(result, da.Array) and np.isnan(result.shape[0]) and converted_chunks == []
        computed = result.compute()
        assert converted_chunks and source.data is source_data and not source._in_memory

        # Compare point rows independently of the raster chunk order
        expected_order = np.lexsort((expected[:, 0], expected[:, 1]))
        computed_order = np.lexsort((computed[:, 0], computed[:, 1]))
        np.testing.assert_array_equal(expected[expected_order], computed[computed_order])

    @pytest.mark.parametrize("as_array", [False, True])
    def test_to_pointcloud__dask_large_empty_sample(self, as_array: bool, tmp_path: Path) -> None:
        """Checks that a large Dask sample with no valid cells returns an empty lazy result."""

        import dask.array as da

        # Write an all-nodata raster whose requested sample is larger than one input chunk
        source_file = tmp_path / "dask-empty-source.tif"
        values = np.ma.masked_all((20, 20), dtype=np.int16)
        values.data.fill(-9999)
        gu.Raster.from_array(values, rio.transform.from_origin(0, 20, 1, 1), 32633, nodata=-9999).to_file(source_file)
        source = open_raster(str(source_file), chunks={"x": 5, "y": 5})

        # Keep the empty output lazy and leave the source raster unloaded after explicit computation
        result = source.rst.to_pointcloud(subsample=0.5, random_state=42, as_array=as_array)
        if as_array:
            assert isinstance(result, da.Array)
            assert result.compute().shape == (0, 3)
        else:
            assert is_dask_dataframe(result)
            assert result.compute().empty
        assert not source._in_memory

    def test_to_pointcloud__multiprocessing_reads_each_selected_tile_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a large sample is selected and read one raster tile at a time."""

        from geoutils.interface import raster_point
        from geoutils.multiproc import readers

        # Write unique values across six tiles so the selected points can be checked independently of output order
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "tiled-point-source.tif"
        output_file = tmp_path / "tiled-points.gpkg"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633)
        raster.to_file(source_file)
        source = gu.Raster(source_file)
        chunks = (3, 4)

        # Record the cells passed to every multi-band read while preserving the real file operation
        read_groups: list[NDArray[np.int64]] = []
        read_selected = readers._read_selected_raster_bands

        def count_selected_tiles(source_raster: Any, indexes: Any, bands: list[int], tile_size: Any) -> Any:
            """Record one selected cell group before reading its raster values."""

            read_groups.append(np.asarray(indexes, dtype=np.int64))
            return read_selected(source_raster, indexes, bands, tile_size)

        monkeypatch.setattr(readers, "_read_selected_raster_bands", count_selected_tiles)

        # Reject the small-sample path because 17 points exceed the largest 3 x 4 tile
        def reject_indexed_path(*args: Any, **kwargs: Any) -> Any:
            """Fail if point conversion collects and groups the complete sample indexes."""

            raise AssertionError("Large point samples must use the bounded cutoff path.")

        monkeypatch.setattr(raster_point, "_raster_to_pointcloud_from_indices", reject_indexed_path)

        # Convert a spatially scattered sample through the synchronous multiprocessing interface
        result = source.to_pointcloud(
            subsample=17,
            random_state=42,
            mp_config=MultiprocConfig(chunks=chunks, outfile=str(output_file)),
        )

        # Check that every group belongs to one tile and that no selected tile is read a second time
        tile_columns = (source.width + chunks[1] - 1) // chunks[1]
        grouped_tile_ids = []
        for indexes in read_groups:
            rows, columns = np.unravel_index(indexes, source.shape)
            tile_ids = (rows // chunks[0]) * tile_columns + columns // chunks[1]
            assert len(np.unique(tile_ids)) == 1
            grouped_tile_ids.append(int(tile_ids[0]))
        assert len(grouped_tile_ids) == len(set(grouped_tile_ids))

        # Confirm the returned point cloud stays unopened until its values are requested for comparison
        assert not source.is_loaded and not result.is_loaded
        expected = raster.subsample(17, random_state=42, strategy="topk")
        np.testing.assert_array_equal(np.sort(result.ds["b1"].to_numpy()), np.sort(expected))
        assert not source.is_loaded

    @pytest.mark.parametrize("skip_nodata", [False, True])
    def test_to_pointcloud__multiprocessing_filters_large_sample_before_auxiliary_reads(
        self, skip_nodata: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a large sample reads auxiliary bands only for cells selected by the cutoff."""

        from geoutils.multiproc import readers

        # Write three bands whose 17 selected cells exceed the largest 3 x 4 tile
        main_values = np.arange(80, dtype=np.int16).reshape((8, 10))
        values = np.stack((main_values, main_values + 100, main_values + 200))
        source_file = tmp_path / f"large-multiband-source-{skip_nodata}.tif"
        output_file = tmp_path / f"large-multiband-points-{skip_nodata}.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)
        source = gu.Raster(source_file)

        # Record the indexes and bands passed to each raster read after the cutoff is known
        grouped_reads = []
        read_selected = readers._read_selected_raster_bands

        def record_grouped_read(source_raster: Any, indexes: Any, bands: list[int], chunks: Any) -> Any:
            """Record cells and bands from one raster read."""

            grouped_reads.append((np.asarray(indexes, dtype=np.int64), bands))
            return read_selected(source_raster, indexes, bands, chunks)

        monkeypatch.setattr(readers, "_read_selected_raster_bands", record_grouped_read)

        # Convert a deterministic sample while requesting all three bands
        result = source.to_pointcloud(
            subsample=17,
            skip_nodata=skip_nodata,
            random_state=42,
            auxiliary_data_bands=[2, 3],
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Pass only the 17 selected indexes to the multiband reader when every cell is eligible
        if not skip_nodata:
            assert sum(len(indexes) for indexes, bands in grouped_reads if bands == [1, 2, 3]) == 17

        # Read the main band to establish eligibility, then pass only 17 indexes to the auxiliary reader
        else:
            assert sum(len(indexes) for indexes, bands in grouped_reads if bands == [2, 3]) == 17
        assert result.point_count == 17
        assert not source.is_loaded and not result.is_loaded

    def test_to_pointcloud__multiprocessing_small_sample_uses_indexed_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a sample no larger than one raster chunk uses the shared indexed conversion path."""

        from geoutils.interface import raster_point

        # Write a raster whose 11 selected cells fit within the largest 3 x 4 chunk
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "small-sample-source.tif"
        output_file = tmp_path / "small-sample-points.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)
        source = gu.Raster(source_file)

        # Record the shared indexed conversion while preserving its result
        indexed_conversion = raster_point._raster_to_pointcloud_from_indices
        indexed_calls = []

        def record_indexed_conversion(*args: Any, **kwargs: Any) -> Any:
            """Record whether the indexed conversion builds an array or point cloud."""

            indexed_calls.append(kwargs["as_array"])
            return indexed_conversion(*args, **kwargs)

        monkeypatch.setattr(raster_point, "_raster_to_pointcloud_from_indices", record_indexed_conversion)

        # Convert the bounded sample and write its materialized point rows once
        result = source.to_pointcloud(
            subsample=11,
            random_state=42,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Return the requested file as an unloaded point cloud after using the shared indexed path
        assert indexed_calls == [False]
        assert output_file.exists()
        assert result.name == str(output_file)
        assert not source.is_loaded and not result.is_loaded

    def test_to_pointcloud__multiprocessing_small_sample_groups_multiband_reads(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that a small sample reads all requested bands once per selected raster tile."""

        from geoutils.multiproc import readers

        # Write three bands whose 11 selected cells fit within one 3 x 4 chunk of output memory
        main_values = np.arange(80, dtype=np.int16).reshape((8, 10))
        values = np.stack((main_values, main_values + 100, main_values + 200))
        source_file = tmp_path / "small-multiband-source.tif"
        output_file = tmp_path / "small-multiband-points.gpkg"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633).to_file(source_file)
        source = gu.Raster(source_file)

        # Record each grouped multi-band file read while preserving its values
        grouped_reads = []
        read_selected = readers._read_selected_raster_bands

        def record_grouped_read(source_raster: Any, indexes: Any, bands: list[int], chunks: Any) -> Any:
            """Record the selected cells and bands read in one worker call."""

            grouped_reads.append((np.asarray(indexes, dtype=np.int64), bands))
            return read_selected(source_raster, indexes, bands, chunks)

        monkeypatch.setattr(readers, "_read_selected_raster_bands", record_grouped_read)

        # Convert all three bands through the indexed multiprocessing path
        result = source.to_pointcloud(
            subsample=11,
            random_state=42,
            auxiliary_data_bands=[2, 3],
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Read each selected cell once, with all bands together and no tile split across calls
        assert sum(len(indexes) for indexes, _ in grouped_reads) == 11
        assert all(bands == [1, 2, 3] for _, bands in grouped_reads)
        tile_columns = (source.width + 3) // 4
        read_tile_ids = []
        for indexes, _ in grouped_reads:
            rows, columns = np.unravel_index(indexes, source.shape)
            tile_ids = (rows // 3) * tile_columns + columns // 4
            assert len(np.unique(tile_ids)) == 1
            read_tile_ids.append(int(tile_ids[0]))
        assert len(read_tile_ids) == len(set(read_tile_ids))
        assert not result.is_loaded

    def test_to_pointcloud__loaded_multiprocessing_array_uses_eager_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that array output does not send a loaded raster through multiprocessing tasks."""

        from geoutils.raster import base

        # Record the configuration passed from to_pointcloud() into the shared sampling implementation
        sampling_configs = []
        subsample = base._subsample

        def record_sampling_config(*args: Any, **kwargs: Any) -> Any:
            """Record the worker configuration before selecting raster cells."""

            sampling_configs.append(kwargs["mp_config"])
            return subsample(*args, **kwargs)

        monkeypatch.setattr(base, "_subsample", record_sampling_config)

        # Request multiprocessing for an array result from values that are already in memory
        values = np.arange(20).reshape((4, 5))
        source = gu.Raster.from_array(values, rio.transform.from_origin(0, 4, 1, 1), 32633)
        result = source.to_pointcloud(
            subsample=7,
            random_state=42,
            as_array=True,
            mp_config=MultiprocConfig(chunks=(2, 3)),
        )

        # Select directly from the loaded array without passing a multiprocessing configuration to subsample()
        assert sampling_configs == [None]
        assert result.shape == (7, 3)

    def test_to_pointcloud__multiprocessing_batches_small_file_writes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Checks that point rows from several small raster tiles are written in one GeoPackage batch."""

        from geoutils.pointcloud import writing

        # Write a source whose scattered sample reaches several 3 x 4 raster tiles
        values = np.arange(80, dtype=np.int16).reshape((8, 10))
        source_file = tmp_path / "batched-point-source.tif"
        output_file = tmp_path / "batched-points.gpkg"
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 8, 1, 1), 32633)
        raster.to_file(source_file)
        source = gu.Raster(source_file)

        # Record the number of rows in each final GeoPackage write
        written_rows = []
        write_dataframe = writing.pyogrio.write_dataframe

        def record_write(dataframe: Any, *args: Any, **kwargs: Any) -> Any:
            """Record one file batch before writing its rows."""

            written_rows.append(len(dataframe))
            return write_dataframe(dataframe, *args, **kwargs)

        monkeypatch.setattr(writing.pyogrio, "write_dataframe", record_write)

        # Convert 17 scattered cells, which is well below the bounded file-write batch size
        result = source.to_pointcloud(
            subsample=17,
            random_state=42,
            mp_config=MultiprocConfig(chunks=(3, 4), outfile=str(output_file)),
        )

        # Check that all selected rows were written together and both wrappers remain unloaded
        assert written_rows == [17]
        assert result.point_count == 17
        assert not source.is_loaded and not result.is_loaded

    def test_to_pointcloud__multiprocessing_empty_output_stays_unloaded(self, tmp_path: Path) -> None:
        """Checks that an all-nodata raster produces an empty file-backed PointCloud without loading its source."""

        # Write an all-nodata raster split into several uneven tiles
        source_file = tmp_path / "empty-source.tif"
        output_file = tmp_path / "empty-points.gpkg"
        values = np.ma.masked_all((5, 7), dtype=np.int16)
        values.data.fill(-9999)
        gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), 32633, nodata=-9999).to_file(source_file)
        source = gu.Raster(source_file)

        # Let each worker save an empty point partition and assemble their common GeoPackage schema
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
        """Checks that to_pointcloud() rejects multiprocessing with Dask without loading the source."""

        import dask.array as da

        # Build a lazy Dask raster that will trigger Dask execution
        values = da.arange(20, chunks=7).reshape((4, 5))
        transform = rio.transform.from_origin(0, 4, 1, 1)
        source = gu.RasterAccessor.from_array(values, transform, 4326)
        source_array = source.data

        # Reject multiprocessing before computing or replacing the source Dask array
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            source.rst.to_pointcloud(as_array=True, mp_config=MultiprocConfig(chunks=(2, 3)))
        assert source.data is source_array
        assert not source._in_memory

    def test_to_pointcloud__error_xarray_with_multiproc(self) -> None:
        """Checks that to_pointcloud() rejects multiprocessing for a non-Dask Xarray accessor."""

        # Build an eager Xarray raster without a Dask chunk layout
        values = np.arange(20).reshape((4, 5))
        source = gu.RasterAccessor.from_array(values, rio.transform.from_origin(0, 4, 1, 1), 4326)

        # Require a Raster input rather than silently writing the Xarray values to a temporary raster
        with pytest.raises(ValueError, match="requires a Raster input"):
            source.rst.to_pointcloud(mp_config=MultiprocConfig(chunks=(2, 3)))
