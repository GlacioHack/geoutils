"""Tests for point cloud reprojection across eager, Dask and multiprocessing backends."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster


@pytest.mark.filterwarnings("ignore:Overriding 3D points with with data column 'intensity':UserWarning")
class TestReprojectChunked:
    """
    Checks reproject() for point clouds.

    - Eager, Dask and Multiproc outputs have the same coordinates and attributes, with file results not loaded.
    - LAS, LAZ and GeoPackage outputs preserve the values their formats can represent.
    - Empty inputs, in-place calls and invalid output choices are checked separately.
    """

    # Give heights values distinct from active intensity values so writing the wrong quantity as LAS Z is visible
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
        Checks that every backend returns the same coordinates and attributes.

        File inputs and the Multiproc output are not loaded.
        """

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")

        # 1/ Prepare independent eager and file sources with an incomplete final row chunk
        # Eleven points split unevenly for both chunk sizes, exposing lost or duplicated rows at chunk edges
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

        # 2/ Reproject each interface to the same neighboring UTM zone
        # GeoPandas provides an independent coordinate reference and preserves the original geometry heights
        target_crs = CRS.from_epsg(32632)
        expected = self.points.to_crs(target_crs)
        eager_result = source.reproject(crs=target_crs)
        accessor_result = accessor.pc.reproject(crs=target_crs)
        lazy_result = lazy.pc.reproject(crs=target_crs)
        outfile = tmp_path / "projected.gpkg"
        with MpCluster({"nb_workers": 2}) as cluster:
            configuration = MultiprocConfig(chunks=chunks, outfile=str(outfile), cluster=cluster)
            multiproc_result = multiproc.reproject(crs=target_crs, mp_config=configuration)

        # 3/ Check metadata and loading before reading the partitioned results
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

        # 4/ Compare complete rows, including order, attributes and geometry Z after output files are closed
        # GeoPackage can normalize integer widths, so compare attribute values without requiring identical dtypes
        computed_lazy = lazy_result.compute()
        for frame in (eager_result.ds, accessor_result, computed_lazy, multiproc_result.ds):
            assert_geodataframe_equal(frame, expected, check_dtype=False)
            np.testing.assert_array_equal(frame.geometry.z, self.heights)
            np.testing.assert_array_equal(frame["intensity"], self.points["intensity"])

        # Reading an output must not load its original source or replace the Dask graph with eager values
        assert multiproc.is_loaded == loaded
        assert not lazy.pc.is_loaded and not lazy_result.pc.is_loaded
        assert source.crs == multiproc.crs == lazy.pc.crs == self.points.crs

    @pytest.mark.parametrize("source_suffix,output_suffix", [(".gpkg", ".las"), (".las", ".laz")])
    def test_reproject__las_coordinates_and_attributes(
        self, source_suffix: str, output_suffix: str, tmp_path: Path
    ) -> None:
        """Checks that LAS and LAZ output contain the same elevations and attributes for any active value column."""

        laspy = import_optional("laspy")
        if output_suffix == ".laz":
            import_optional("lazrs")

        # Write either 3D vector geometry or native LAS elevations independently of the reprojection implementation
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

        # Project uneven row chunks while retaining the active intensity column and unloaded source
        source = gu.PointCloud(filename, data_column="intensity")
        expected = self.points.to_crs(32632)
        outfile = tmp_path / ("projected" + output_suffix)
        with MpCluster({"nb_workers": 2}) as cluster:
            configuration = MultiprocConfig(chunks=4, outfile=str(outfile), cluster=cluster)
            result = source.reproject(crs=32632, mp_config=configuration)
        assert not source.is_loaded and not result.is_loaded
        assert result.data_column == "intensity"
        assert result.crs == expected.crs and result.point_count == len(expected)

        # Read the written LAS records independently and allow only the file's coordinate quantization error
        records = laspy.read(outfile)
        tolerance = records.header.scales / 2 + 1e-8
        np.testing.assert_allclose(records.x, expected.geometry.x, rtol=0, atol=tolerance[0])
        np.testing.assert_allclose(records.y, expected.geometry.y, rtol=0, atol=tolerance[1])
        np.testing.assert_allclose(records.z, self.heights, rtol=0, atol=tolerance[2])
        np.testing.assert_array_equal(records.intensity, self.points["intensity"])
        np.testing.assert_array_equal(records.quality, self.points["quality"])
        np.testing.assert_array_equal(records.row_id, self.positions)
        assert records.header.parse_crs() == expected.crs

        # Loading active values uses intensity rather than the independent elevations stored in native LAS Z
        np.testing.assert_array_equal(result.data, self.points["intensity"])
        assert not source.is_loaded

    @pytest.mark.parametrize("target_crs", [32633, 4326])
    def test_reproject__reference_and_default_output_format(self, target_crs: int, tmp_path: Path) -> None:
        """Checks that reference reprojection writes an unloaded GeoPackage even when the target CRS is unchanged."""

        # Use a separate reference object and file-backed source so neither needs a full read to choose its CRS
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="intensity")
        expected = self.points.to_crs(target_crs)
        reference = gu.Vector(expected)
        outfile = tmp_path / "projected"

        # Infer GeoPackage for an output path without a suffix and inspect its unloaded result metadata
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))
        result = source.reproject(ref=reference, mp_config=configuration)
        assert outfile.exists()
        assert result is not source
        assert not result.is_loaded and not source.is_loaded
        assert result.crs == reference.crs

        # Reuse the extensionless file as an unloaded source for another projection before inspecting its data
        second_configuration = MultiprocConfig(chunks=4, outfile=str(tmp_path / "reprojected.gpkg"))
        second_result = result.reproject(crs=self.points.crs, mp_config=second_configuration)
        assert not result.is_loaded and not second_result.is_loaded
        assert_geodataframe_equal(second_result.ds, expected.to_crs(self.points.crs), check_dtype=False)
        assert not result.is_loaded

        # Both written outputs agree with the equivalent GeoPandas transformations, leaving the source unloaded
        assert_geodataframe_equal(result.ds, expected, check_dtype=False)
        assert not source.is_loaded

    @pytest.mark.parametrize("loaded", [False, True])
    def test_reproject__empty_point_cloud(self, loaded: bool, tmp_path: Path) -> None:
        """Checks that empty inputs write point files with the same attributes and CRS without loading them."""

        # Write a typed empty point layer so file metadata identifies its geometry without any point records
        frame = self.points.iloc[:0].copy()
        filename = tmp_path / "empty.gpkg"
        frame.to_file(filename, geometry_type="Point", index=False)
        source = gu.PointCloud(frame, data_column="intensity") if loaded else gu.PointCloud(filename, "intensity")
        expected = frame.to_crs(32632)
        configuration = MultiprocConfig(chunks=4, outfile=str(tmp_path / "projected.gpkg"))

        # Reprojection must write an empty point schema instead of skipping output or inventing a placeholder row
        result = source.reproject(crs=32632, mp_config=configuration)
        assert not result.is_loaded and source.is_loaded == loaded
        assert result.point_count == 0
        assert result.crs == expected.crs and result.data_column == "intensity"
        assert list(result.columns) == list(frame.columns)
        assert not result.is_loaded

        # Reading the output returns every empty attribute column while the original source is not loaded
        assert_geodataframe_equal(result.ds, expected, check_dtype=False)
        assert source.is_loaded == loaded

    def test_reproject__error_inplace_with_chunked_execution(self, tmp_path: Path) -> None:
        """Checks that multiprocessing rejects in-place replacement while eager reprojection still supports it."""

        # Use an unloaded MP source so rejection happens before its data or the output file are touched
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="intensity")
        outfile = tmp_path / "projected.gpkg"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))
        with pytest.raises(ValueError, match="inplace|in place"):
            source.reproject(crs=32632, inplace=True, mp_config=configuration)
        assert not source.is_loaded and not outfile.exists()
        assert source.crs == self.points.crs

        # Without multiprocessing, the same public option updates the source and returns None
        eager = gu.PointCloud(self.points.copy(), data_column="intensity")
        expected = self.points.to_crs(32632)
        assert eager.reproject(crs=32632, inplace=True) is None
        assert_geodataframe_equal(eager.ds, expected)

    def test_reproject__multiprocessing_accessor_dataframe_output(self, tmp_path: Path) -> None:
        """Checks that multiprocessing through a GeoDataFrame accessor returns the same dataframe family."""

        # Store active values in a named column separate from the source's three-dimensional geometry
        source = self.points.copy()
        source.pc.set_data_column("intensity")
        expected = self.points.to_crs(32632)
        configuration = MultiprocConfig(chunks=4, outfile=str(tmp_path / "projected.gpkg"))

        # The accessor reads the completed point file into a GeoDataFrame with the original active column
        result = source.pc.reproject(crs=32632, mp_config=configuration)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.pc.data_column == "intensity"
        assert_geodataframe_equal(result, expected, check_dtype=False)
        assert source.pc.crs == self.points.crs

    @pytest.mark.parametrize("data_column", ["intensity", None])
    def test_reproject__las_accessor_attributes_and_active_values(
        self, data_column: str | None, tmp_path: Path
    ) -> None:
        """Checks that LAS accessor output contains every input attribute and selects intensity or native Z."""

        laspy = import_optional("laspy")

        # Select either an auxiliary attribute or geometry heights without removing the other point values
        source = self.points.copy()
        source.pc.set_data_column(data_column)
        outfile = tmp_path / "projected.las"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))
        result = source.pc.reproject(crs=32632, mp_config=configuration)

        # LAS stores geometry heights in its native Z column, which becomes active when geometry was selected
        assert isinstance(result, gpd.GeoDataFrame)
        expected_column = "Z" if data_column is None else data_column
        assert result.pc.data_column == expected_column
        for column in ("intensity", "quality", "row_id"):
            np.testing.assert_array_equal(result[column], self.points[column])

        # Allow only the written Z scale's rounding error and preserve the caller's original active selection
        with laspy.open(outfile) as reader:
            tolerance = reader.header.scales[2] / 2 + 1e-8
        expected_values = self.heights if data_column is None else self.points["intensity"]
        np.testing.assert_allclose(result["Z"], self.heights, rtol=0, atol=tolerance)
        np.testing.assert_allclose(result.pc.data, expected_values, rtol=0, atol=tolerance)
        assert source.pc.data_column == data_column

    def test_reproject__error_dask_with_multiprocessing(self, tmp_path: Path) -> None:
        """Checks that Dask and multiprocessing cannot be combined or execute partitions before rejection."""

        import_optional("dask_geopandas", package_name="dask-geopandas")
        from dask.callbacks import Callback

        # Build a file-backed Dask source with several row partitions
        filename = tmp_path / "source.gpkg"
        self.points.to_file(filename, index=False)
        source = gu.open_pointcloud(str(filename), data_column="intensity", chunks=4)
        outfile = tmp_path / "projected.gpkg"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))

        # Reject competing execution backends before computing any of the source partitions
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
        """Checks that unsupported output formats and raster-shaped chunks fail before reading the source."""

        # Use a valid point source and vary only the output option that is incompatible with point reprojection
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

        # No output should be created and the input metadata must still describe the original unloaded file
        with pytest.raises(ValueError):
            source.reproject(crs=32632, mp_config=configuration)
        assert not outfile.exists() and not source.is_loaded
        assert source.crs == self.points.crs

    def test_reproject__error_las_without_elevations(self, tmp_path: Path) -> None:
        """Checks that LAS output rejects two-dimensional points whose active values do not define elevations."""

        import_optional("laspy")

        # Remove geometry heights but leave intensity present; it must not silently become the output LAS Z
        frame = gpd.GeoDataFrame(
            self.points.drop(columns="geometry"),
            geometry=gpd.points_from_xy(self.points.geometry.x, self.points.geometry.y),
            crs=self.points.crs,
        )
        source = gu.PointCloud(frame, data_column="intensity")
        outfile = tmp_path / "projected.las"
        configuration = MultiprocConfig(chunks=4, outfile=str(outfile))

        # Require a geometry height or native LAS Z column before a point file can be written
        with pytest.raises(ValueError, match="elevation|Z|3D"):
            source.reproject(crs=32632, mp_config=configuration)
        assert not outfile.exists()
        np.testing.assert_array_equal(source.data, self.points["intensity"])

    @pytest.mark.parametrize(
        "attribute_kind",
        ["nullable_integer", "submillisecond_datetime", "fractional_intensity", "out_of_range_intensity"],
    )
    def test_reproject__error_lossy_attribute_storage(self, attribute_kind: str, tmp_path: Path) -> None:
        """Checks that unrepresentable attribute values raise before replacing an existing output file."""

        # Give three points values that the destination format would otherwise round or wrap without an error
        frame = self.points.iloc[:3].copy()
        if attribute_kind == "nullable_integer":
            # Integer columns with nodata can force a float conversion that cannot represent values above 2**53 exactly
            column, suffix = "identifier", ".gpkg"
            frame[column] = pd.Series([2**53 + 1, pd.NA, 2**53 + 3], dtype="Int64")
        elif attribute_kind == "submillisecond_datetime":
            # GeoPackage datetime storage cannot represent these nanoseconds below the millisecond boundary
            column, suffix = "observed_at", ".gpkg"
            frame[column] = pd.date_range("2024-01-01T00:00:00.123456789", periods=3, freq="s")
        else:
            import_optional("laspy")
            column, suffix = "intensity", ".las"
            if attribute_kind == "fractional_intensity":
                frame[column] = np.array([0.5, 1.5, 2.5])
            else:
                frame[column] = np.array([0, 65536, 70000], dtype=np.uint32)

        # Stage one row per worker task to expose conversions that depend on an individual row's nodata value
        source = gu.PointCloud(frame, data_column="intensity")
        original_values = frame[column].copy()
        outfile = tmp_path / ("projected" + suffix)
        original_output = b"original"
        outfile.write_bytes(original_output)
        configuration = MultiprocConfig(chunks=1, outfile=str(outfile))

        # A failed encoding must identify its attribute and leave both the existing output and source unchanged
        with pytest.raises(ValueError, match=column):
            source.reproject(crs=32632, mp_config=configuration)
        assert outfile.read_bytes() == original_output
        pd.testing.assert_series_equal(source.ds[column], original_values)

    @pytest.mark.parametrize("attribute_kind", ["nullable_integer", "millisecond_datetime"])
    def test_reproject__gpkg_representable_attributes(self, attribute_kind: str, tmp_path: Path) -> None:
        """Checks that GeoPackage output contains exact small integers and millisecond datetime values."""

        # Use values representable by the file format so precision checks do not reject valid point attributes
        frame = self.points.iloc[:3].copy()
        if attribute_kind == "nullable_integer":
            column = "identifier"
            frame[column] = pd.Series([1, pd.NA, 3], dtype="Int64")
        else:
            column = "observed_at"
            frame[column] = pd.date_range("2024-01-01T00:00:00.123", periods=3, freq="ms")
        source = gu.PointCloud(frame, data_column="intensity")
        configuration = MultiprocConfig(chunks=1, outfile=str(tmp_path / "projected.gpkg"))

        # Use single-row chunks and place a nodata value in the middle chunk to check schema consistency
        result = source.reproject(crs=32632, mp_config=configuration)
        assert not result.is_loaded
        actual = result.ds[column]
        expected = frame[column]

        # Compare value precision independently of the reader's integer-null or datetime dtype representation
        if attribute_kind == "nullable_integer":
            actual_values = actual.to_numpy(dtype=float, na_value=np.nan)
            expected_values = expected.to_numpy(dtype=float, na_value=np.nan)
            np.testing.assert_array_equal(actual_values, expected_values)
        else:
            np.testing.assert_array_equal(actual.to_numpy(dtype="datetime64[ns]"), expected.to_numpy())
