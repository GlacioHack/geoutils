"""Tests on the Pandas accessor mirroring the Vector API."""

from __future__ import annotations

import os.path
import tempfile
from importlib.util import find_spec
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import box

import geoutils as gu
import geoutils.vector.pd_accessor as pd_accessor
from geoutils._misc import import_optional


class TestVectorAccessor:
    """Check that the Pandas ``vct`` accessor mirrors Vector for eager and lazy dataframes."""

    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    def test_open_vector(self) -> None:
        """Open a vector file as the same eager GeoDataFrame used by Vector."""

        ds = gu.open_vector(self.aster_outlines_path, load=True)

        assert isinstance(ds, gpd.GeoDataFrame)
        assert_geodataframe_equal(ds, gu.Vector(self.aster_outlines_path).gdf)

    def test_bbox__dask_geopandas(self) -> None:
        """Checks that a lazy vector reads its bounding box without replacing its Dask collection."""

        # Open the same features through lazy and eager GeoPandas representations
        pytest.importorskip("dask_geopandas")
        lazy = gu.open_vector(self.aster_outlines_path, chunks=1)
        eager = gu.open_vector(self.aster_outlines_path, load=True)

        # Compute the total bounding box while keeping the accessor backed by Dask
        assert lazy.vct.bbox == rio.coords.BoundingBox(*eager.total_bounds)
        assert not lazy.vct.is_loaded

    def test_equality__dask_partitions(self) -> None:
        """Compare lazy vectors exactly and approximately without changing their collections."""

        dgpd = pytest.importorskip("dask_geopandas")
        import dask

        eager = gu.open_vector(self.aster_outlines_path, load=True)
        lazy = gu.open_vector(self.aster_outlines_path, chunks=1)
        with dask.config.set({"dataframe.convert-string": False}):
            differently_chunked = dgpd.from_geopandas(eager, npartitions=2)
        close = eager.copy()
        close.geometry = close.geometry.translate(xoff=1e-9)
        changed = eager.copy()
        changed.geometry = changed.geometry.translate(xoff=1)

        assert lazy.vct.vector_equal(eager)
        assert lazy.vct.vector_equal(differently_chunked)
        assert lazy.vct.vector_allclose(close, atol=1e-8)
        assert not lazy.vct.vector_equal(close)
        assert not lazy.vct.vector_equal(changed)
        assert not lazy.vct.is_loaded

    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("crop", {"bbox": (-180, -90, 180, 90)}),
            ("reproject", {"crs": 4326}),
            ("translate", {"xoff": 1, "yoff": 2}),
            ("query", {"expression": "index == index"}),
        ],
    )
    def test_inplace__dask_rejected(self, method: str, kwargs: dict[str, object]) -> None:
        """Explain that immutable Dask vectors must use the returned collection."""

        pytest.importorskip("dask_geopandas")
        ds = gu.open_vector(self.aster_outlines_path, chunks=1)

        with pytest.raises(ValueError, match="cannot be modified in place"):
            getattr(ds.vct, method)(**kwargs, inplace=True)
        assert not ds.vct.is_loaded

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

    def test_copy_and_bounds(self) -> None:
        """Copy the dataframe independently while preserving its vector bounds."""

        ds = gu.open_vector(self.aster_outlines_path, load=True)
        ds_copy = ds.vct.copy()

        assert ds_copy is not ds
        assert ds.vct.bounds == gu.Vector(ds).bounds
        assert_geodataframe_equal(ds_copy, ds)

    @pytest.mark.parametrize("as_pointcloud", [False, True])
    @pytest.mark.parametrize(
        "method,options",
        [
            ("copy", {}),
            ("crop", {"bbox": (-1, -1, 3, 3)}),
            ("clip", {"mask": (-1, -1, 3, 3)}),
            ("reproject", {"crs": 4326}),
            ("translate", {"xoff": 1, "yoff": 2}),
            ("query", {"expression": "height > 1"}),
        ],
    )
    def test_methods__metadata(self, method: str, options: dict[str, Any], as_pointcloud: bool) -> None:
        """Checks that operations preserve custom metadata."""

        dgpd = import_optional("dask_geopandas", package_name="dask-geopandas")
        from dask.callbacks import Callback

        from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

        # Create dataframe with custom metadata
        coordinates = np.arange(5, dtype=float)
        geometries = gpd.points_from_xy(coordinates, coordinates)
        if not as_pointcloud:
            geometries = geometries.buffer(0.25)
        frame = gpd.GeoDataFrame({"height": coordinates}, geometry=geometries, crs=32631)
        frame.attrs["title"] = "survey features"
        accessor = "pc" if as_pointcloud else "vct"
        pd_accessor._register_dask_vector_accessor()
        if as_pointcloud:
            _register_dask_pointcloud_accessor()
            frame.pc.set_data_name("height")
        source = dgpd.from_geopandas(frame, chunksize=2, sort=False)

        # Run each operation and check metadata is unchanged
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            result = getattr(getattr(source, accessor), method)(**options)
        assert not tasks
        assert not getattr(source, accessor).is_loaded and not getattr(result, accessor).is_loaded
        expected = getattr(getattr(frame, accessor), method)(**options)
        actual = result.compute()
        assert_geodataframe_equal(actual, expected)
        assert actual.attrs["title"] == "survey features"
        assert source._meta.attrs["title"] == "survey features"
        assert not getattr(source, accessor).is_loaded and not getattr(result, accessor).is_loaded

    def test_to_file(self) -> None:
        """Write an eager accessor-backed dataframe through the Vector-compatible API."""

        ds = gu.open_vector(self.aster_outlines_path, load=True)

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
        assert gu.Vector(temp_file).vector_equal(
            gu.Vector(gu.open_vector(self.aster_outlines_path, load=True).to_crs(3857))
        )
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
        assert_geodataframe_equal(gpd.read_parquet(temp_file), gu.open_vector(self.aster_outlines_path, load=True))
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
        assert point_mask.pc.data_name == "z"

        proximity = ds.vct.proximity(size=(5, 5))
        assert isinstance(proximity, xr.DataArray)

    def test_create_mask_points__dask_geopandas(self) -> None:
        """Checks that create_mask() creates lazy Dask-GeoPandas and Dask array point masks."""

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
        dask_points = gu.open_pointcloud(temp_file, data_name="id", chunks=2, as_object_type="geodataframe")

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
