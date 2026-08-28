"""Tests on the Pandas accessor mirroring the Vector API."""

from __future__ import annotations

import os.path
import tempfile
from importlib.util import find_spec

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import box

import geoutils as gu
import geoutils.vector.pd_accessor as pd_accessor


class TestVectorAccessor:
    """Check that the Pandas ``vct`` accessor mirrors Vector for eager and lazy dataframes."""

    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    def test_open_vector(self) -> None:
        """Open a vector file as the same eager GeoDataFrame used by Vector."""

        ds = gu.open_vector(self.aster_outlines_path)

        assert isinstance(ds, gpd.GeoDataFrame)
        assert_geodataframe_equal(ds, gu.Vector(self.aster_outlines_path).ds)

    def test_open_vector__dask_geopandas(self) -> None:
        """Open and reproject vector partitions without eagerly loading them."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Request one feature per partition to exercise the lazy open path
        ds = gu.open_vector(self.aster_outlines_path, chunks=1)

        assert isinstance(ds, dgpd.GeoDataFrame)
        assert not ds.vct.is_loaded

        # Build the expected result eagerly while Dask keeps its operation deferred
        reprojected = ds.vct.reproject(crs=CRS.from_epsg(3857))
        expected = gu.open_vector(self.aster_outlines_path).to_crs(CRS.from_epsg(3857))

        # Compute only for the final geometry and CRS comparison
        assert isinstance(reprojected, dgpd.GeoDataFrame)
        assert not reprojected.vct.is_loaded
        assert_geodataframe_equal(reprojected.compute(), expected)
        assert not ds.vct.is_loaded
        assert not reprojected.vct.is_loaded

    def test_translate_vector__dask_geopandas(self) -> None:
        """Translate vector partitions lazily and match the eager accessor result."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Apply the same offset through lazy and eager accessors
        ds = gu.open_vector(self.aster_outlines_path, chunks=1)
        translated = ds.vct.translate(xoff=1, yoff=2)
        expected = gu.open_vector(self.aster_outlines_path).vct.translate(xoff=1, yoff=2)

        # Translation must preserve laziness on both source and output
        assert isinstance(translated, dgpd.GeoDataFrame)
        assert not ds.vct.is_loaded
        assert not translated.vct.is_loaded
        assert_geodataframe_equal(translated.compute(), expected)
        assert not ds.vct.is_loaded
        assert not translated.vct.is_loaded

    @pytest.mark.parametrize(
        ("method", "clip"),
        [("copy", False), ("crop", False), ("crop", True)],
        ids=["copy", "crop", "crop-and-clip"],
    )
    def test_copy_crop_vector__dask_geopandas(self, method: str, clip: bool) -> None:
        """Keep copied and cropped vector partitions lazy and equal to eager GeoPandas."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Use the middle half of the source extent so cropping has visible work to perform
        expected_source = gu.open_vector(self.aster_outlines_path)
        left, bottom, right, top = expected_source.total_bounds
        bbox = (
            left + (right - left) / 4,
            bottom + (top - bottom) / 4,
            right - (right - left) / 4,
            top - (top - bottom) / 4,
        )
        kwargs = {"bbox": bbox, "clip": clip} if method == "crop" else {}

        # Apply the same operation to eager and lazy accessors
        ds = gu.open_vector(self.aster_outlines_path, chunks=1)
        expected = getattr(expected_source.vct, method)(**kwargs)
        output = getattr(ds.vct, method)(**kwargs)

        assert isinstance(output, dgpd.GeoDataFrame)
        assert not ds.vct.is_loaded
        assert not output.vct.is_loaded
        computed_output = output.compute()

        # Partition order is not meaningful, but every feature and its original index must remain exact
        assert_geodataframe_equal(computed_output.sort_index(), expected.sort_index())
        assert not ds.vct.is_loaded
        assert not output.vct.is_loaded

    def test_open_vector__dask_geopandas_missing_dep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explain the missing optional dependency when chunked vector opening requests Dask."""

        import_optional = pd_accessor.import_optional

        def _missing_dask_geopandas(
            import_name: str, package_name: str | None = None, extra_name: str = "opt"
        ) -> object:
            """Fail only the Dask-GeoPandas import while forwarding other imports."""

            if import_name == "dask_geopandas":
                raise ImportError("Optional dependency 'dask-geopandas' required.")
            return import_optional(import_name, package_name=package_name, extra_name=extra_name)

        # Replace the module-level optional importer used by ``open_vector``
        monkeypatch.setattr(pd_accessor, "import_optional", _missing_dask_geopandas)
        with pytest.raises(ImportError, match="Optional dependency 'dask-geopandas' required.*"):
            gu.open_vector(self.aster_outlines_path, chunks=1)

    def test_to_geoutils(self) -> None:
        """Convert an accessor-backed dataframe to an equivalent Vector."""

        ds = gu.open_vector(self.aster_outlines_path)
        vector = ds.vct.to_geoutils()

        assert isinstance(vector, gu.Vector)
        assert vector.vector_equal(gu.Vector(self.aster_outlines_path))

    def test_copy_and_bounds(self) -> None:
        """Copy the dataframe independently while preserving its vector bounds."""

        ds = gu.open_vector(self.aster_outlines_path)
        ds_copy = ds.vct.copy()

        assert ds_copy is not ds
        assert ds.vct.bounds == gu.Vector(ds).bounds
        assert_geodataframe_equal(ds_copy, ds)

    def test_methods(self) -> None:
        """Match Vector outputs for common crop, reproject and translate methods."""

        # Wrap the same GeoDataFrame through both public interfaces
        ds = gu.open_vector(self.aster_outlines_path)
        vector = gu.Vector(ds)

        # Compare each operation independently to make failures easy to locate
        cropped_ds = ds.vct.crop(ds.vct.bounds)
        cropped_vector = vector.crop(vector.bounds)
        assert_geodataframe_equal(cropped_ds, cropped_vector.ds)

        reproj_ds = ds.vct.reproject(crs=CRS.from_epsg(4326))
        reproj_vector = vector.reproject(crs=CRS.from_epsg(4326))
        assert_geodataframe_equal(reproj_ds, reproj_vector.ds)

        translated_ds = ds.vct.translate(xoff=1, yoff=2)
        translated_vector = vector.translate(xoff=1, yoff=2)
        assert_geodataframe_equal(translated_ds, translated_vector.ds)

    def test_to_file(self) -> None:
        """Write an eager accessor-backed dataframe through the Vector-compatible API."""

        ds = gu.open_vector(self.aster_outlines_path)

        # Use a temporary GeoPackage and reopen it through Vector for comparison
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        ds.vct.to_file(temp_file)

        assert os.path.exists(temp_file)
        assert gu.Vector(temp_file).vector_equal(gu.Vector(ds))

    @pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
    def test_to_file__dask_geopandas_gpkg(self) -> None:
        """Compute lazy vector partitions while writing a single GeoPackage."""

        # Include a lazy reprojection so writing evaluates the complete task graph
        ds = gu.open_vector(self.aster_outlines_path, chunks=1).vct.reproject(crs=CRS.from_epsg(3857))
        assert not ds.vct.is_loaded

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        ds.vct.to_file(temp_file)

        # Reopen the file eagerly and compare it with the projected source
        assert os.path.exists(temp_file)
        assert gu.Vector(temp_file).vector_equal(gu.Vector(gu.open_vector(self.aster_outlines_path).to_crs(3857)))
        assert not ds.vct.is_loaded

    @pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask-geopandas is installed.")
    def test_to_file__dask_geopandas_parquet(self) -> None:
        """Write lazy vector partitions to GeoParquet without changing features."""

        pytest.importorskip("pyarrow")

        # GeoParquet supports a partitioned Dask write directly
        ds = gu.open_vector(self.aster_outlines_path, chunks=1)
        assert not ds.vct.is_loaded

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.parquet")
        ds.vct.to_file(temp_file)

        # Read through GeoPandas to validate the complete written dataset
        assert os.path.exists(temp_file)
        assert_geodataframe_equal(gpd.read_parquet(temp_file), gu.open_vector(self.aster_outlines_path))
        assert not ds.vct.is_loaded

    def test_cross_type_outputs_are_accessors(self) -> None:
        """Return accessor-native objects when vector operations change geospatial type."""

        # One polygon is enough to exercise raster, mask and point-cloud outputs
        ds = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs=4326)

        # Raster-producing methods should return Xarray DataArrays
        raster = ds.vct.rasterize(res=0.5)
        assert isinstance(raster, xr.DataArray)

        raster_mask = ds.vct.create_mask(res=0.5)
        assert isinstance(raster_mask, xr.DataArray)

        # Point masks should return a GeoDataFrame carrying the ``pc`` accessor
        point_mask = ds.vct.create_mask(points=(np.array([0.5, 2.0]), np.array([0.5, 2.0])))
        assert isinstance(point_mask, gpd.GeoDataFrame)
        assert point_mask.pc.data_column == "z"

        proximity = ds.vct.proximity(size=(5, 5))
        assert isinstance(proximity, xr.DataArray)

    def test_create_mask_points__dask_geopandas(self) -> None:
        """Create lazy point masks as either Dask-GeoPandas or Dask Array output."""

        dgpd = pytest.importorskip("dask_geopandas")
        import dask.array as da

        # Define points inside, above and outside a simple polygon
        vector = gpd.GeoDataFrame({"geometry": [box(0, 0, 2, 2)]}, crs=3857)
        points = gpd.GeoDataFrame(
            {"id": [1, 2, 3]},
            geometry=gpd.points_from_xy([0.5, 1.5, 3.0], [0.5, 2.5, 1.0]),
            crs=3857,
        )

        # Open the points in two partitions to exercise the Dask mask path
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "points.gpkg")
        points.to_file(temp_file)
        dask_points = gu.open_pointcloud(temp_file, data_column="id", chunks=2)

        # Compare lazy point-cloud output with the eager mask
        expected = vector.vct.create_mask(points=points)
        output = vector.vct.create_mask(points=dask_points)

        assert isinstance(output, dgpd.GeoDataFrame)
        assert not output.pc.is_loaded
        assert_geodataframe_equal(output.compute(), expected)
        assert not dask_points.pc.is_loaded
        assert not output.pc.is_loaded

        # Requesting an array should preserve the same partitioned values
        output_array = vector.vct.create_mask(points=dask_points, as_array=True)
        assert isinstance(output_array, da.Array)
        assert np.array_equal(output_array.compute(), expected["z"].to_numpy())
        assert not dask_points.pc.is_loaded

        # The generic ``ref`` argument must dispatch to the same point-cloud path
        output_ref = vector.vct.create_mask(ref=dask_points, as_array=True)
        assert isinstance(output_ref, da.Array)
        assert np.array_equal(output_ref.compute(), expected["z"].to_numpy())
        assert not dask_points.pc.is_loaded
