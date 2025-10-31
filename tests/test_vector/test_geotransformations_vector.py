"""Tests for geotransformations of vectors."""

from __future__ import annotations

import re

import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

import geoutils as gu


class TestGeotransformations:

    landsat_b4_path = gu.examples.get_path_test("everest_landsat_b4")
    landsat_b4_crop_path = gu.examples.get_path_test("everest_landsat_b4_cropped")
    everest_outlines_path = gu.examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = gu.examples.get_path_test("exploradores_aster_dem")
    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    def test_reproject(self) -> None:
        """Test that the reproject function works as intended"""

        v0 = gu.Vector(self.aster_outlines_path)
        r0 = gu.Raster(self.aster_dem_path)
        v1 = gu.Vector(self.everest_outlines_path)

        # First, test with a EPSG integer
        v1 = v0.reproject(crs=32617)
        assert isinstance(v1, gu.Vector)
        assert v1.crs.to_epsg() == 32617

        # Check the inplace behaviour matches the not-inplace one
        v2 = v0.copy()
        v2.reproject(crs=32617, inplace=True)
        v2.vector_equal(v1)

        # Check that the reprojection is the same as with geopandas
        gpd1 = v0.ds.to_crs(epsg=32617)
        assert_geodataframe_equal(gpd1, v1.ds)

        # Second, with a Raster object
        v2 = v0.reproject(r0)
        assert v2.crs == r0.crs

        # Third, with a Vector object that has a different CRS
        assert v0.crs != v1.crs
        v3 = v0.reproject(v1)
        assert v3.crs == v1.crs

        # Fourth, check that errors are raised when appropriate
        # When no destination CRS is defined, or both dst_crs and dst_ref are passed
        with pytest.raises(ValueError, match=re.escape("Either of `ref` or `crs` must be set. Not both.")):
            v0.reproject()
            v0.reproject(ref=r0, crs=32617)
        # If input of wrong type
        with pytest.raises(TypeError, match=re.escape("Type of ref must be a raster or vector.")):
            v0.reproject(ref=10)  # type: ignore

    test_data = [[landsat_b4_path, everest_outlines_path], [aster_dem_path, aster_outlines_path]]

    @pytest.mark.parametrize("data", test_data)  # type: ignore
    def test_crop(self, data: list[str]) -> None:
        # Load data
        raster_path, outlines_path = data
        rst = gu.Raster(raster_path)
        outlines = gu.Vector(outlines_path)

        # Need to reproject to r.crs. Otherwise, crop will work but will be approximate
        # Because outlines might be warped in a different crs
        outlines.ds = outlines.ds.to_crs(rst.crs)

        # Crop
        outlines_new = outlines.copy()
        outlines_new.crop(crop_geom=rst, inplace=True)

        # Check default behaviour - crop and return copy
        outlines_copy = outlines.crop(crop_geom=rst)

        # Crop by passing bounds
        outlines_new_bounds = outlines.copy()
        outlines_new_bounds.crop(crop_geom=list(rst.bounds), inplace=True)
        assert_geodataframe_equal(outlines_new.ds, outlines_new_bounds.ds)
        # Check the return-by-copy as well
        assert_geodataframe_equal(outlines_copy.ds, outlines_new_bounds.ds)

        # Verify that geometries intersect with raster bound
        rst_poly = gu.projtools.bounds2poly(rst.bounds)
        intersects_new = []
        for poly in outlines_new.ds.geometry:
            intersects_new.append(poly.intersects(rst_poly))

        assert np.all(intersects_new)

        # Check that some of the original outlines did not intersect and were removed
        intersects_old = []
        for poly in outlines.ds.geometry:
            intersects_old.append(poly.intersects(rst_poly))

        assert np.sum(intersects_old) == np.sum(intersects_new)

        # Check that some features were indeed removed if any geometry didn't intersect the raster bounds
        if any(~np.array(intersects_old)):
            assert np.sum(~np.array(intersects_old)) > 0

        # Check that error is raised when cropGeom argument is invalid
        with pytest.raises(TypeError, match="Crop geometry must be a Raster, Vector, or list of coordinates."):
            outlines.crop(1, inplace=True)  # type: ignore

    def test_translate(self) -> None:

        vector = gu.Vector(self.everest_outlines_path)

        # Check default behaviour is not inplace
        vector_shifted = vector.translate(xoff=2.5, yoff=5.7)
        assert isinstance(vector_shifted, gu.Vector)
        assert_geoseries_equal(vector_shifted.geometry, vector.geometry.translate(xoff=2.5, yoff=5.7))

        # Check inplace behaviour works correctly
        vector2 = vector.copy()
        output = vector2.translate(xoff=2.5, yoff=5.7, inplace=True)
        assert output is None
        assert_geoseries_equal(vector2.geometry, vector_shifted.geometry)
