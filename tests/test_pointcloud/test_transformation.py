"""Tests for point cloud transformations."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import Polygon

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster


class TestTransformation:
    """Test module for cropping and clipping point clouds."""

    def test_crop_and_clip__different_point_selection(self) -> None:
        """Checks that crop() uses a bounding box and clip() uses the given shape."""

        # Create two points inside the triangle's bounding box, with one point outside the triangle
        points = gpd.GeoDataFrame(
            {"value": [1.0, 2.0]},
            geometry=gpd.points_from_xy([0.25, 1.75], [0.25, 1.75]),
            crs=32610,
        )
        pointcloud = gu.PointCloud(points, data_column="value")
        triangle = Polygon([(0, 0), (2, 0), (0, 2)])

        # Compare the points selected by the bounding box and the triangle
        cropped = pointcloud.crop(triangle.bounds)
        clipped = pointcloud.clip(triangle)
        assert cropped.point_count == 2
        assert clipped.point_count == 1
        assert clipped["value"].iloc[0] == 1.0

        # Check that the old crop option warns and gives the same PointCloud result
        with pytest.warns(DeprecationWarning, match="Argument 'clip' is deprecated"):
            deprecated = pointcloud.crop((0, 0, 1, 1), clip=True)
        assert isinstance(deprecated, gu.PointCloud)
        assert deprecated.point_count == 1
        assert deprecated["value"].iloc[0] == 1.0

    def test_crop__deferred_file_read(self, tmp_path: Path) -> None:
        """Checks that crop() does not read points from a file until they are needed."""

        # Write four points with two on each side of the crop boundary
        points = gpd.GeoDataFrame(
            {"value": [1.0, 2.0, 3.0, 4.0]},
            geometry=gpd.points_from_xy([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]),
            crs=32610,
        )
        path = tmp_path / "points.gpkg"
        points.to_file(path)
        pointcloud = gu.PointCloud(path, data_column="value")

        # Select the two points on the left and keep both objects unloaded
        cropped = pointcloud.crop((-0.5, -0.5, 0.5, 1.5))
        assert not pointcloud.is_loaded
        assert not cropped.is_loaded

        # Read the result and compare it with the known points on the left
        expected = gu.PointCloud(points.iloc[[0, 2]].reset_index(drop=True), data_column="value")
        assert cropped.pointcloud_equal(expected)
        assert not pointcloud.is_loaded


class TestTransformationChunked:
    """Test module for point cloud transformations run with Dask or multiprocessing."""

    clip_positions = np.arange(11)
    clip_points = gpd.GeoDataFrame(
        {
            "intensity": (100 + clip_positions).astype(np.int32),
            "quality": clip_positions / 16,
            "row_id": clip_positions.astype(np.int32),
        },
        geometry=gpd.points_from_xy(clip_positions, (3 * clip_positions) % 8, 20 + clip_positions / 8),
        crs=32610,
    )
    clip_geometry = Polygon([(0, 0), (10, 0), (0, 10)])

    def test_clip__chunked_backends_equal(self, tmp_path: Path) -> None:
        """Checks that Dask and multiprocessing clip() select the same points as an in-memory call."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Write eleven 3D points so the final group contains fewer than four points
        filename = tmp_path / "points_to_clip.gpkg"
        self.clip_points.to_file(filename, index=False)

        # Clip the same points in memory, with Dask and with two worker processes
        expected = gu.PointCloud(self.clip_points, data_column="intensity").clip(self.clip_geometry).ds
        expected = expected.sort_values("row_id").reset_index(drop=True)
        lazy = gu.open_pointcloud(str(filename), data_column="intensity", chunks=4)
        multiproc = gu.PointCloud(filename, data_column="intensity")
        lazy_result = lazy.pc.clip(self.clip_geometry)
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=4, outfile=str(tmp_path / "points_clipped.gpkg"), cluster=cluster)
            multiproc_result = multiproc.clip(self.clip_geometry, mp_config=config)

        # Check that Dask and file results have the expected types and remain unloaded
        assert isinstance(lazy_result, dgpd.GeoDataFrame)
        assert not lazy.pc.is_loaded and not lazy_result.pc.is_loaded
        assert isinstance(multiproc_result, gu.PointCloud)
        assert not multiproc.is_loaded and not multiproc_result.is_loaded
        assert multiproc_result.data_column == "intensity"
        assert multiproc_result.point_count == len(expected)
        with pytest.raises(ValueError, match="cannot be combined with a Dask point cloud"):
            lazy.pc.clip(self.clip_geometry, mp_config=config)

        # Read the results and compare every selected point and value
        computed_lazy = lazy_result.compute().sort_values("row_id").reset_index(drop=True)
        computed_multiproc = multiproc_result.ds.sort_values("row_id").reset_index(drop=True)
        assert_geodataframe_equal(computed_lazy, expected, check_dtype=False)
        assert_geodataframe_equal(computed_multiproc, expected, check_dtype=False)
        assert not multiproc.is_loaded
        assert not lazy.pc.is_loaded and not lazy_result.pc.is_loaded

    def test_clip__dask_las_loading_laziness(self) -> None:
        """Checks that clip() keeps a LAS-backed Dask GeoDataFrame lazy and matches eager point selection."""

        pytest.importorskip("laspy")
        dgpd = pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # Derive a mask from LAS header bounds without reading the point records
        filename = gu.examples.get_path_test("coromandel_lidar")
        eager = gu.PointCloud(filename)
        left, bottom, right, top = eager.bbox
        middle = left + (right - left) / 2
        geometry = Polygon([(left, bottom), (middle, bottom), (middle, top), (left, top)])
        assert not eager.is_loaded

        # Build one clipping task per 100-point LAS partition without running any Dask task
        lazy = gu.open_pointcloud(filename, chunks=100)
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            lazy_result = lazy.pc.clip(geometry)
        assert tasks == []
        assert isinstance(lazy_result, dgpd.GeoDataFrame)
        assert lazy_result.npartitions == lazy.npartitions
        assert lazy_result.pc.data_column == lazy.pc.data_column == "Z"
        assert not lazy.pc.is_loaded and not lazy_result.pc.is_loaded

        # Compute only the result and compare the same points independently of Dask partition order
        expected = eager.clip(geometry).ds
        expected = expected.assign(_x=expected.geometry.x, _y=expected.geometry.y)
        expected = expected.sort_values(["_x", "_y", "Z"]).drop(columns=["_x", "_y"]).reset_index(drop=True)
        computed = lazy_result.compute()
        computed = computed.assign(_x=computed.geometry.x, _y=computed.geometry.y)
        computed = computed.sort_values(["_x", "_y", "Z"]).drop(columns=["_x", "_y"]).reset_index(drop=True)
        assert_geodataframe_equal(computed, expected, check_dtype=False)
        assert not lazy.pc.is_loaded and not lazy_result.pc.is_loaded

    def test_clip__multiprocessing_las_output(self, tmp_path: Path) -> None:
        """Checks that multiprocessing clip() writes the selected points and values to a LAS file."""

        laspy = pytest.importorskip("laspy")

        # Write the 3D points to a file that can be read in groups
        filename = tmp_path / "points_to_las.gpkg"
        self.clip_points.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="intensity")
        expected = self.clip_points.clip(self.clip_geometry).sort_values("row_id")

        # Clip groups of four points and write the result to one LAS file
        outfile = tmp_path / "points_clipped.las"
        config = MultiprocConfig(chunks=4, outfile=str(outfile))
        result = source.clip(self.clip_geometry, mp_config=config)
        assert not source.is_loaded and not result.is_loaded
        assert result.point_count == len(expected)
        assert result.data_column == "intensity"

        # Compare the stored coordinates and point values with the expected result
        records = laspy.read(outfile)
        order = np.argsort(records.row_id)
        tolerance = records.header.scales / 2 + 1e-8
        np.testing.assert_allclose(np.asarray(records.x)[order], expected.geometry.x, rtol=0, atol=tolerance[0])
        np.testing.assert_allclose(np.asarray(records.y)[order], expected.geometry.y, rtol=0, atol=tolerance[1])
        np.testing.assert_allclose(np.asarray(records.z)[order], expected.geometry.z, rtol=0, atol=tolerance[2])
        np.testing.assert_array_equal(np.asarray(records.intensity)[order], expected["intensity"])
        np.testing.assert_array_equal(np.asarray(records.quality)[order], expected["quality"])
        np.testing.assert_array_equal(np.asarray(records.row_id)[order], expected["row_id"])

    # Use different height and intensity values, so the test catches intensity being written to LAS Z by mistake
    positions = np.arange(11)
    heights = 20 + positions / 8
    points = gpd.GeoDataFrame(
        {
            "intensity": (100 + positions).astype(np.int32),
            "quality": positions / 16,
            "row_id": positions.astype(np.int32),
        },
        geometry=gpd.points_from_xy(500000 + 3 * positions, 5100000 + 2 * positions, heights),
        crs=32633,
    )

    @pytest.mark.parametrize("chunks", [4, 6])
    @pytest.mark.parametrize("loaded", [False, True])
    def test_reproject__chunked_backends_equal(self, chunks: int, loaded: bool, tmp_path: Path) -> None:
        """
        Checks that eager, Dask and multiprocessing reprojection return the same rows and metadata.

        File-backed inputs keep their original loaded/unloaded state, and the new file output stays unloaded.
        """

        dgpd = pytest.importorskip("dask_geopandas")

        # 1/ Prepare eager, accessor, Dask and multiprocessing inputs from the same 11 points
        # Chunks of 4 or 6 both leave a shorter final chunk, which helps catch dropped/duplicated rows at the joins
        filename = tmp_path / "points.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.PointCloud(self.points.copy(), data_column="intensity")
        accessor = self.points.copy()
        accessor.pc.set_data_column("intensity")
        lazy = gu.open_pointcloud(str(filename), data_column="intensity", chunks=chunks)
        multiproc = gu.PointCloud(filename, data_column="intensity")
        if loaded:
            multiproc.load()
        assert multiproc.is_loaded == loaded
        assert not lazy.pc.is_loaded

        # 2/ Reproject every input to the neighboring UTM zone
        # Use GeoPandas for the expected X/Y coordinates; its result also keeps the original Z heights
        target_crs = CRS.from_epsg(32632)
        expected = self.points.to_crs(target_crs)
        eager_result = source.reproject(crs=target_crs)
        accessor_result = accessor.pc.reproject(crs=target_crs)
        lazy_result = lazy.pc.reproject(crs=target_crs)
        outfile = tmp_path / "projected.gpkg"
        with MpCluster({"nb_workers": 2}) as cluster:
            configuration = MultiprocConfig(chunks=chunks, outfile=str(outfile), cluster=cluster)
            multiproc_result = multiproc.reproject(crs=target_crs, mp_config=configuration)

        # 3/ Check the output types, metadata and loaded state before reading any partitioned result
        assert isinstance(eager_result, gu.PointCloud) and eager_result.is_loaded
        assert isinstance(accessor_result, gpd.GeoDataFrame)
        assert isinstance(lazy_result, dgpd.GeoDataFrame) and not lazy_result.pc.is_loaded
        assert isinstance(multiproc_result, gu.PointCloud) and not multiproc_result.is_loaded
        assert multiproc_result.crs == target_crs
        assert multiproc_result.point_count == len(self.points)
        assert multiproc_result.data_column == "intensity"
        np.testing.assert_allclose(multiproc_result.bounds, expected.total_bounds, rtol=0, atol=1e-9)
        assert not multiproc_result.is_loaded
        assert multiproc.is_loaded == loaded

        # 4/ Read each result and compare all rows (same order, attributes, X/Y and Z)
        # GeoPackage may change an integer width, so equal values are enough even if the dtypes differ
        computed_lazy = lazy_result.compute()
        for frame in (eager_result.ds, accessor_result, computed_lazy, multiproc_result.ds):
            assert_geodataframe_equal(frame, expected, check_dtype=False)
            np.testing.assert_array_equal(frame.geometry.z, self.heights)
            np.testing.assert_array_equal(frame["intensity"], self.points["intensity"])

        # Reading the outputs must not change whether the original file source was loaded
        # The Dask input and output also remain Dask dataframes after lazy_result.compute()
        assert multiproc.is_loaded == loaded
        assert not lazy.pc.is_loaded and not lazy_result.pc.is_loaded
        assert source.crs == multiproc.crs == lazy.pc.crs == self.points.crs

    @pytest.mark.parametrize("source_suffix,output_suffix", [(".gpkg", ".las"), (".las", ".laz")])
    def test_reproject__las_coordinates_and_attributes(
        self, source_suffix: str, output_suffix: str, tmp_path: Path
    ) -> None:
        """Checks that LAS/LAZ keep elevations and attributes when intensity is the active value column."""

        laspy = pytest.importorskip("laspy")
        if output_suffix == ".laz":
            pytest.importorskip("lazrs")

        # Create both kinds of input used here: GeoPackage keeps height in geometry, while LAS keeps it in Z
        filename = tmp_path / ("source" + source_suffix)
        if source_suffix == ".gpkg":
            self.points.to_file(filename, index=False)
        else:
            header = laspy.LasHeader(point_format=6, version="1.4")
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.array([500000.0, 5100000.0, 0.0])
            header.add_crs(self.points.crs)
            header.add_extra_dim(laspy.ExtraBytesParams(name="quality", type=np.float64))
            header.add_extra_dim(laspy.ExtraBytesParams(name="row_id", type=np.int32))
            records = laspy.LasData(header)
            records.x = self.points.geometry.x.to_numpy()
            records.y = self.points.geometry.y.to_numpy()
            records.z = self.heights
            records.intensity = self.points["intensity"].to_numpy()
            records.quality, records.row_id = self.points["quality"].to_numpy(), self.positions
            records.write(filename)

        # Reproject the 11 points in chunks of four and keep intensity active without loading the source object
        source = gu.PointCloud(filename, data_column="intensity")
        expected = self.points.to_crs(32632)
        outfile = tmp_path / ("projected" + output_suffix)
        with MpCluster({"nb_workers": 2}) as cluster:
            configuration = MultiprocConfig(chunks=4, outfile=str(outfile), cluster=cluster)
            result = source.reproject(crs=32632, mp_config=configuration)
        assert not source.is_loaded and not result.is_loaded
        assert result.data_column == "intensity"
        assert result.crs == expected.crs and result.point_count == len(expected)

        # Open the written file with laspy and compare X/Y/Z within half of the LAS storage scale
        # Other attributes are stored exactly and should match without a tolerance
        records = laspy.read(outfile)
        tolerance = records.header.scales / 2 + 1e-8
        np.testing.assert_allclose(records.x, expected.geometry.x, rtol=0, atol=tolerance[0])
        np.testing.assert_allclose(records.y, expected.geometry.y, rtol=0, atol=tolerance[1])
        np.testing.assert_allclose(records.z, self.heights, rtol=0, atol=tolerance[2])
        np.testing.assert_array_equal(records.intensity, self.points["intensity"])
        np.testing.assert_array_equal(records.quality, self.points["quality"])
        np.testing.assert_array_equal(records.row_id, self.positions)
        assert records.header.parse_crs() == expected.crs

        # Loading PointCloud.data returns intensity, the separate height values remain in the LAS Z field
        np.testing.assert_array_equal(result.data, self.points["intensity"])
        assert not source.is_loaded

    @pytest.mark.parametrize("target_crs", [32633, 4326])
    def test_reproject__reference_and_default_output_format(self, target_crs: int, tmp_path: Path) -> None:
        """Checks that ref= sets the CRS and an output without a suffix becomes an unloaded GeoPackage."""

        # Write the points once, then make a reference object in the requested CRS (including the unchanged CRS case)
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="intensity")
        expected = self.points.to_crs(target_crs)
        reference = gu.Vector(expected)
        outfile = tmp_path / "projected"

        # Leave off the output suffix; reprojection should create a GeoPackage
        # Its CRS and other metadata should be available without loading any rows
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))
        result = source.reproject(ref=reference, mp_config=configuration)
        assert outfile.exists()
        assert result is not source
        assert not result.is_loaded and not source.is_loaded
        assert result.crs == reference.crs

        # Use that first output as an unloaded source and project it back to the starting CRS
        second_configuration = MultiprocConfig(chunks=4, outfile=str(tmp_path / "reprojected.gpkg"))
        second_result = result.reproject(crs=self.points.crs, mp_config=second_configuration)
        assert not result.is_loaded and not second_result.is_loaded
        assert_geodataframe_equal(second_result.ds, expected.to_crs(self.points.crs), check_dtype=False)
        assert not result.is_loaded

        # Both files match the same GeoPandas CRS changes, and reading them did not load the original source
        assert_geodataframe_equal(result.ds, expected, check_dtype=False)
        assert not source.is_loaded

    @pytest.mark.parametrize("loaded", [False, True])
    def test_reproject__empty_point_cloud(self, loaded: bool, tmp_path: Path) -> None:
        """Checks that an empty output keeps its columns/CRS and a file source stays unloaded."""

        # Write zero rows but keep the point geometry type, columns and CRS in the file metadata
        frame = self.points.iloc[:0].copy()
        filename = tmp_path / "empty.gpkg"
        frame.to_file(filename, geometry_type="Point", index=False)
        source = gu.PointCloud(frame, data_column="intensity") if loaded else gu.PointCloud(filename, "intensity")
        expected = frame.to_crs(32632)
        configuration = MultiprocConfig(chunks=4, outfile=str(tmp_path / "projected.gpkg"))

        # Reproject the empty input; the output still needs a valid point schema and must not add a placeholder row
        result = source.reproject(crs=32632, mp_config=configuration)
        assert not result.is_loaded and source.is_loaded == loaded
        assert result.point_count == 0
        assert result.crs == expected.crs and result.data_column == "intensity"
        assert list(result.columns) == list(frame.columns)
        assert not result.is_loaded

        # Reading the result gives the expected empty columns and does not change the source's loaded state
        assert_geodataframe_equal(result.ds, expected, check_dtype=False)
        assert source.is_loaded == loaded

    def test_reproject__multiprocessing_accessor_dataframe_output(self, tmp_path: Path) -> None:
        """Checks that multiprocessing through the accessor returns a GeoDataFrame with intensity still active."""

        # Make intensity the active values while the point heights remain in the 3D geometry
        source = self.points.copy()
        source.pc.set_data_column("intensity")
        expected = self.points.to_crs(32632)
        configuration = MultiprocConfig(chunks=4, outfile=str(tmp_path / "projected.gpkg"))

        # The accessor reads the written file back into a GeoDataFrame and keeps intensity as the data column
        result = source.pc.reproject(crs=32632, mp_config=configuration)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.pc.data_column == "intensity"
        assert_geodataframe_equal(result, expected, check_dtype=False)
        assert source.pc.crs == self.points.crs

    @pytest.mark.parametrize("data_column", ["intensity", None])
    def test_reproject__las_accessor_attributes_and_active_values(
        self, data_column: str | None, tmp_path: Path
    ) -> None:
        """Checks that LAS accessor output keeps all attributes and makes intensity or LAS Z active."""

        laspy = pytest.importorskip("laspy")

        # Select intensity, or select geometry height with None; the other point attributes should still be written
        source = self.points.copy()
        source.pc.set_data_column(data_column)
        outfile = tmp_path / "projected.las"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))
        result = source.pc.reproject(crs=32632, mp_config=configuration)

        # LAS stores geometry height in Z, so Z becomes active when data_column=None selected the geometry values
        assert isinstance(result, gpd.GeoDataFrame)
        expected_column = "Z" if data_column is None else data_column
        assert result.pc.data_column == expected_column
        for column in ("intensity", "quality", "row_id"):
            np.testing.assert_array_equal(result[column], self.points[column])

        # LAS rounds Z to its storage scale, so allow half a scale step when checking heights/active values
        # The source dataframe must still have the data column chosen above
        with laspy.open(outfile) as reader:
            tolerance = reader.header.scales[2] / 2 + 1e-8
        expected_values = self.heights if data_column is None else self.points["intensity"]
        np.testing.assert_allclose(result["Z"], self.heights, rtol=0, atol=tolerance)
        np.testing.assert_allclose(result.pc.data, expected_values, rtol=0, atol=tolerance)
        assert source.pc.data_column == data_column

    @pytest.mark.parametrize("attribute_kind", ["nullable_integer", "millisecond_datetime"])
    def test_reproject__gpkg_representable_attributes(self, attribute_kind: str, tmp_path: Path) -> None:
        """Checks that GeoPackage keeps small integers and millisecond datetimes exactly."""

        # Use small integers and times rounded to milliseconds, because GeoPackage can store both exactly
        frame = self.points.iloc[:3].copy()
        if attribute_kind == "nullable_integer":
            column = "identifier"
            frame[column] = pd.Series([1, pd.NA, 3], dtype="Int64")
        else:
            column = "observed_at"
            frame[column] = pd.date_range("2024-01-01T00:00:00.123", periods=3, freq="ms")
        source = gu.PointCloud(frame, data_column="intensity")
        configuration = MultiprocConfig(chunks=1, outfile=str(tmp_path / "projected.gpkg"))

        # Write one row per chunk; the nullable integer case puts nodata alone in the middle chunk
        # This checks that all chunks still use one compatible output column type
        result = source.reproject(crs=32632, mp_config=configuration)
        assert not result.is_loaded
        actual = result.ds[column]
        expected = frame[column]

        # The file reader may choose another dtype, so convert both sides before comparing the stored values
        if attribute_kind == "nullable_integer":
            actual_values = actual.to_numpy(dtype=float, na_value=np.nan)
            expected_values = expected.to_numpy(dtype=float, na_value=np.nan)
            np.testing.assert_array_equal(actual_values, expected_values)
        else:
            np.testing.assert_array_equal(actual.to_numpy(dtype="datetime64[ns]"), expected.to_numpy())


class TestReprojectErrors:
    """Test module for validation errors raised by eager, Dask, and multiprocessing reprojection."""

    points = TestTransformationChunked.points
    heights = TestTransformationChunked.heights

    def test_reproject__error_inplace_with_chunked_execution(self, tmp_path: Path) -> None:
        """Checks that multiprocessing rejects inplace before writing, while eager reprojection updates the object."""

        # Start with an unloaded file source and a new output path, then request inplace + multiprocessing
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="intensity")
        outfile = tmp_path / "projected.gpkg"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))
        with pytest.raises(ValueError, match="inplace|in place"):
            source.reproject(crs=32632, inplace=True, mp_config=configuration)
        assert not source.is_loaded and not outfile.exists()
        assert source.crs == self.points.crs

        # The eager version accepts inplace, returns None and changes the source coordinates
        eager = gu.PointCloud(self.points.copy(), data_column="intensity")
        expected = self.points.to_crs(32632)
        assert eager.reproject(crs=32632, inplace=True) is None
        assert_geodataframe_equal(eager.ds, expected)

    def test_reproject__error_dask_with_multiprocessing(self, tmp_path: Path) -> None:
        """Checks that Dask + multiprocessing is rejected before any Dask partition runs."""

        pytest.importorskip("dask_geopandas")
        from dask.callbacks import Callback

        # Open the file as a Dask dataframe split into several row partitions
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.open_pointcloud(str(filename), data_column="intensity", chunks=4)
        outfile = tmp_path / "projected.gpkg"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))

        # Request multiprocessing too, and record any Dask task that runs before the expected error
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(args[0])):
            with pytest.raises(ValueError, match="Dask"):
                source.pc.reproject(crs=32632, mp_config=configuration)
        assert tasks == []
        assert not source.pc.is_loaded and not outfile.exists()

    @pytest.mark.parametrize(
        "invalid_option", ["driver", "suffix", "driver_suffix", "chunks", "extensionless_las", "extensionless_laz"]
    )
    def test_reproject__error_invalid_output_options(self, invalid_option: str, tmp_path: Path) -> None:
        """Checks that bad formats and 2D chunk sizes fail before the point source is read."""

        # Start with one valid point file, then change one output setting to an unsupported choice
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="intensity")
        outfile = tmp_path / ("projected.tif" if invalid_option == "suffix" else "projected.gpkg")
        if invalid_option.startswith("extensionless"):
            outfile = outfile.with_suffix("")
        driver = {
            "driver": "GTiff",
            "driver_suffix": "LAS",
            "extensionless_las": "LAS",
            "extensionless_laz": "LAZ",
        }.get(invalid_option)
        configuration = MultiprocConfig(
            chunks=(4, 4) if invalid_option == "chunks" else 4,
            outfile=str(outfile),
            driver=driver,
        )

        # The error happens before writing or loading rows, and the source CRS stays unchanged
        with pytest.raises(ValueError):
            source.reproject(crs=32632, mp_config=configuration)
        assert not outfile.exists() and not source.is_loaded
        assert source.crs == self.points.crs

    def test_reproject__error_las_without_elevations(self, tmp_path: Path) -> None:
        """Checks that LAS output rejects 2D points even when they have an active intensity column."""

        pytest.importorskip("laspy")

        # Remove Z from the geometry but leave intensity present; intensity must not be used as height by accident
        frame = gpd.GeoDataFrame(
            self.points.drop(columns="geometry"),
            geometry=gpd.points_from_xy(self.points.geometry.x, self.points.geometry.y),
            crs=self.points.crs,
        )
        source = gu.PointCloud(frame, data_column="intensity")
        outfile = tmp_path / "projected.las"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))

        # Writing LAS needs a geometry height or an existing LAS Z column, so no output file should be created
        with pytest.raises(ValueError, match="elevation|Z|3D"):
            source.reproject(crs=32632, mp_config=configuration)
        assert not outfile.exists()
        np.testing.assert_array_equal(source.data, self.points["intensity"])

    @pytest.mark.parametrize(
        "attribute_kind",
        ["nullable_integer", "submillisecond_datetime", "fractional_intensity", "out_of_range_intensity"],
    )
    def test_reproject__error_lossy_attribute_storage(self, attribute_kind: str, tmp_path: Path) -> None:
        """Checks that values a file cannot store exactly raise before an existing output is replaced."""

        # Give three points values that GeoPackage/LAS would have to round, wrap or convert to another value
        frame = self.points.iloc[:3].copy()
        if attribute_kind == "nullable_integer":
            # Nodata can turn an integer column into floats, which cannot store every integer above 2**53 exactly
            column, suffix = "identifier", ".gpkg"
            frame[column] = pd.Series([2**53 + 1, pd.NA, 2**53 + 3], dtype="Int64")
        elif attribute_kind == "submillisecond_datetime":
            # GeoPackage stores milliseconds, so these extra nanoseconds would be lost
            column, suffix = "observed_at", ".gpkg"
            frame[column] = pd.date_range("2024-01-01T00:00:00.123456789", periods=3, freq="s")
        else:
            pytest.importorskip("laspy")
            column, suffix = "intensity", ".las"
            if attribute_kind == "fractional_intensity":
                frame[column] = np.array([0.5, 1.5, 2.5])
            else:
                frame[column] = np.array([0, 65536, 70000], dtype=np.uint32)

        # Send one row to each worker task, so the nodata row is converted in a chunk by itself
        source = gu.PointCloud(frame, data_column="intensity")
        original_values = frame[column].copy()
        outfile = tmp_path / ("projected" + suffix)
        original_output = b"original"
        outfile.write_bytes(original_output)
        configuration = MultiprocConfig(chunks=1, outfile=str(outfile))

        # The error names the bad column, keeps the old output bytes and leaves the source values unchanged
        with pytest.raises(ValueError, match=column):
            source.reproject(crs=32632, mp_config=configuration)
        assert outfile.read_bytes() == original_output
        pd.testing.assert_series_equal(source.ds[column], original_values)
