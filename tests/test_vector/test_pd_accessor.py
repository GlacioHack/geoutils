"""Tests on Pandas accessor mirroring Vector API."""

from __future__ import annotations

import os.path
import tempfile

import geopandas as gpd
import numpy as np
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from pyproj import CRS
from shapely.geometry import box

import geoutils as gu


class TestVectorAccessor:
    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    def test_open_vector(self) -> None:
        ds = gu.open_vector(self.aster_outlines_path)

        assert isinstance(ds, gpd.GeoDataFrame)
        assert_geodataframe_equal(ds, gu.Vector(self.aster_outlines_path).ds)

    def test_to_geoutils(self) -> None:
        ds = gu.open_vector(self.aster_outlines_path)
        vector = ds.vct.to_geoutils()

        assert isinstance(vector, gu.Vector)
        assert vector.vector_equal(gu.Vector(self.aster_outlines_path))

    def test_copy_and_bounds(self) -> None:
        ds = gu.open_vector(self.aster_outlines_path)
        ds_copy = ds.vct.copy()

        assert ds_copy is not ds
        assert ds.vct.bounds == gu.Vector(ds).bounds
        assert_geodataframe_equal(ds_copy, ds)

    def test_methods(self) -> None:
        ds = gu.open_vector(self.aster_outlines_path)
        vector = gu.Vector(ds)

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
        ds = gu.open_vector(self.aster_outlines_path)

        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")
        ds.vct.to_file(temp_file)

        assert os.path.exists(temp_file)
        assert gu.Vector(temp_file).vector_equal(gu.Vector(ds))

    def test_cross_type_outputs_are_accessors(self) -> None:
        ds = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs=4326)

        raster = ds.vct.rasterize(res=0.5)
        assert isinstance(raster, xr.DataArray)

        raster_mask = ds.vct.create_mask(res=0.5)
        assert isinstance(raster_mask, xr.DataArray)

        point_mask = ds.vct.create_mask(points=(np.array([0.5, 2.0]), np.array([0.5, 2.0])))
        assert isinstance(point_mask, gpd.GeoDataFrame)
        assert point_mask.pc.data_column == "z"

        proximity = ds.vct.proximity(size=(5, 5))
        assert isinstance(proximity, xr.DataArray)
