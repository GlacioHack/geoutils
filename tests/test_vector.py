from __future__ import annotations

import inspect
import os.path
import pathlib
import re
import tempfile
import warnings

import geopandas as gpd
import geopandas.base
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from pandas.testing import assert_series_equal
from scipy.ndimage import binary_erosion
from shapely.geometry.base import BaseGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon

import geoutils as gu

GLACIER_OUTLINES_URL = "http://public.data.npolar.no/cryoclim/CryoClim_GAO_SJ_1990.zip"


class TestVector:
    landsat_b4_crop_path = gu.examples.get_path("everest_landsat_b4_cropped")
    everest_outlines_path = gu.examples.get_path("everest_rgi_outlines")
    aster_dem_path = gu.examples.get_path("exploradores_aster_dem")
    aster_outlines_path = gu.examples.get_path("exploradores_rgi_outlines")
    glacier_outlines = gu.Vector(GLACIER_OUTLINES_URL)

    def test_init(self) -> None:
        """Test class initiation works as intended"""

        # First, with a URL filename
        v = gu.Vector(GLACIER_OUTLINES_URL)
        assert isinstance(v, gu.Vector)

        # Second, with a string filename
        v0 = gu.Vector(self.aster_outlines_path)
        assert isinstance(v0, gu.Vector)

        # Third, with a pathlib path
        path = pathlib.Path(self.aster_outlines_path)
        v1 = gu.Vector(path)
        assert isinstance(v1, gu.Vector)

        # Fourth, with a geopandas dataframe
        v2 = gu.Vector(gpd.read_file(self.aster_outlines_path))
        assert isinstance(v2, gu.Vector)

        # Fifth, passing a Vector itself (points back to Vector passed)
        v3 = gu.Vector(v2)
        assert isinstance(v3, gu.Vector)

        # Check errors are raised when filename has wrong type
        with pytest.raises(TypeError, match="Filename argument should be a string, Path or geopandas.GeoDataFrame."):
            gu.Vector(1)  # type: ignore

    def test_copy(self) -> None:
        vector2 = self.glacier_outlines.copy()

        assert vector2 is not self.glacier_outlines

        vector2.ds = vector2.ds.query("NAME == 'Ayerbreen'")

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]

    def test_query(self) -> None:
        vector2 = self.glacier_outlines.query("NAME == 'Ayerbreen'")

        assert vector2 is not self.glacier_outlines

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]

    def test_save(self) -> None:
        """Test the save wrapper for GeoDataFrame.to_file()."""

        vector = gu.Vector(self.aster_outlines_path)

        # Create a temporary file in a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        temp_file = os.path.join(temp_dir.name, "test.gpkg")

        # Save and check the file exists
        vector.save(temp_file)
        assert os.path.exists(temp_file)

        # Open and check the object is the same
        vector_save = gu.Vector(temp_file)
        vector_save.vector_equal(vector)

    def test_bounds(self) -> None:
        bounds = self.glacier_outlines.bounds

        assert bounds.left < bounds.right
        assert bounds.bottom < bounds.top

        assert bounds.left == self.glacier_outlines.ds.total_bounds[0]
        assert bounds.bottom == self.glacier_outlines.ds.total_bounds[1]
        assert bounds.right == self.glacier_outlines.ds.total_bounds[2]
        assert bounds.top == self.glacier_outlines.ds.total_bounds[3]

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
        # If the path provided does not exist
        with pytest.raises(ValueError, match=re.escape("Reference raster or vector path does not exist.")):
            v0.reproject(ref="tmp.lol")
        # If it exists but cannot be opened by rasterio or fiona
        with pytest.raises(ValueError, match=re.escape("Could not open raster or vector with rasterio or fiona.")):
            v0.reproject(ref="geoutils/examples.py")
        # If input of wrong type
        with pytest.raises(TypeError, match=re.escape("Type of ref must be string path to file, Raster or Vector.")):
            v0.reproject(ref=10)  # type: ignore

    def test_rasterize_proj(self) -> None:
        # Capture the warning on resolution not matching exactly bounds
        with pytest.warns(UserWarning):
            burned = self.glacier_outlines.rasterize(xres=3000)

        assert burned.shape[0] == 146
        assert burned.shape[1] == 115

    def test_rasterize_unproj(self) -> None:
        """Test rasterizing an EPSG:3426 dataset into a projection."""

        vct = gu.Vector(self.everest_outlines_path)
        rst = gu.Raster(self.landsat_b4_crop_path)

        # Use Web Mercator at 30 m.
        # Capture the warning on resolution not matching exactly bounds
        with pytest.warns(UserWarning):
            burned = vct.rasterize(xres=30, crs=3857)

        assert burned.shape[0] == 1251
        assert burned.shape[1] == 1522

        # Typically, rasterize returns a raster
        burned_in2_out1 = vct.rasterize(raster=rst, in_value=2, out_value=1)
        assert isinstance(burned_in2_out1, gu.Raster)

        # For an in_value of 1 and out_value of 0 (default), it returns a mask
        burned_mask = vct.rasterize(raster=rst, in_value=1)
        assert isinstance(burned_mask, gu.Mask)

        # Check that rasterizing with in_value=1 is the same as creating a mask
        assert burned_mask.raster_equal(vct.create_mask(raster=rst))

        # The two rasterization should match
        assert np.all(burned_in2_out1[burned_mask] == 2)
        assert np.all(burned_in2_out1[~burned_mask] == 1)

        # Check that errors are raised
        with pytest.raises(ValueError, match="Only one of raster or crs can be provided."):
            vct.rasterize(raster=rst, crs=3857)

    test_data = [[landsat_b4_crop_path, everest_outlines_path], [aster_dem_path, aster_outlines_path]]

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

        # Check that some features were indeed removed
        assert np.sum(~np.array(intersects_old)) > 0

        # Check that error is raised when cropGeom argument is invalid
        with pytest.raises(TypeError, match="Crop geometry must be a Raster, Vector, or list of coordinates."):
            outlines.crop(1, inplace=True)  # type: ignore

    def test_proximity(self) -> None:
        """
        The core functionality is already tested against GDAL in test_raster: just verify the vector-specific behaviour.
        #TODO: add an artificial test as well (mirroring TODO in test_raster)
        """

        vector = gu.Vector(self.everest_outlines_path)

        # -- Test 1: with a Raster provided --
        raster1 = gu.Raster(self.landsat_b4_crop_path)
        prox1 = vector.proximity(raster=raster1)

        # The proximity should have the same extent, resolution and CRS
        assert raster1.georeferenced_grid_equal(prox1)

        # With the base geometry
        vector.proximity(raster=raster1, geometry_type="geometry")

        # With another geometry option
        vector.proximity(raster=raster1, geometry_type="centroid")

        # With only inside proximity
        vector.proximity(raster=raster1, in_or_out="in")

        # -- Test 2: with no Raster provided, just grid size --

        # Default grid size
        vector.proximity()

        # With specific grid size
        vector.proximity(size=(100, 100))


class TestSynthetic:
    # Create a synthetic vector file with a square of size 1, started at position (10, 10)
    poly1 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    gdf = gpd.GeoDataFrame({"geometry": [poly1]}, crs="EPSG:4326")
    vector = gu.Vector(gdf)

    # Same with a square started at position (5, 5)
    poly2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
    gdf = gpd.GeoDataFrame({"geometry": [poly2]}, crs="EPSG:4326")
    vector2 = gu.Vector(gdf)

    # Create a multipolygon with both
    multipoly = MultiPolygon([poly1, poly2])
    gdf = gpd.GeoDataFrame({"geometry": [multipoly]}, crs="EPSG:4326")
    vector_multipoly = gu.Vector(gdf)

    # Create a synthetic vector file with a square of size 5, started at position (8, 8)
    poly3 = Polygon([(8, 8), (13, 8), (13, 13), (8, 13)])
    gdf = gpd.GeoDataFrame({"geometry": [poly3]}, crs="EPSG:4326")
    vector_5 = gu.Vector(gdf)

    # Create a synthetic LineString geometry
    lines = LineString([(10, 10), (11, 10), (11, 11)])
    gdf = gpd.GeoDataFrame({"geometry": [lines]}, crs="EPSG:4326")
    vector_lines = gu.Vector(gdf)

    # Create a synthetic MultiLineString geometry
    multilines = MultiLineString([[(10, 10), (11, 10), (11, 11)], [(5, 5), (6, 5), (6, 6)]])
    gdf = gpd.GeoDataFrame({"geometry": [multilines]}, crs="EPSG:4326")
    vector_multilines = gu.Vector(gdf)

    def test_create_mask(self) -> None:
        """
        Test Vector.create_mask.
        """
        # First with given res and bounds -> Should be a 21 x 21 array with 0 everywhere except center pixel
        vector = self.vector.copy()
        out_mask = vector.create_mask(xres=1, bounds=(0, 0, 21, 21), as_array=True)
        ref_mask = np.zeros((21, 21), dtype="bool")
        ref_mask[10, 10] = True
        assert out_mask.shape == (21, 21)
        assert np.all(ref_mask == out_mask)

        # Check that vector has not been modified by accident
        assert vector.bounds == self.vector.bounds
        assert len(vector.ds) == len(self.vector.ds)
        assert vector.crs == self.vector.crs

        # Then with a gu.Raster as reference, single band
        rst = gu.Raster.from_array(np.zeros((21, 21)), transform=(1.0, 0.0, 0.0, 0.0, -1.0, 21.0), crs="EPSG:4326")
        out_mask = vector.create_mask(rst, as_array=True)
        assert out_mask.shape == (21, 21)

        # With gu.Raster, 2 bands -> fails...
        # rst = gu.Raster.from_array(np.zeros((2, 21, 21)), transform=(1., 0., 0., 0., -1., 21.), crs='EPSG:4326')
        # out_mask = vector.create_mask(rst)

        # Test that buffer = 0 works
        out_mask_buff = vector.create_mask(rst, buffer=0, as_array=True)
        assert np.all(ref_mask == out_mask_buff)

        # Test that buffer > 0 works
        rst = gu.Raster.from_array(np.zeros((21, 21)), transform=(1.0, 0.0, 0.0, 0.0, -1.0, 21.0), crs="EPSG:4326")
        out_mask = vector.create_mask(rst, as_array=True)
        for buffer in np.arange(1, 8):
            out_mask_buff = vector.create_mask(rst, buffer=buffer, as_array=True)
            diff = out_mask_buff & ~out_mask
            assert np.count_nonzero(diff) > 0
            # Difference between masks should always be thinner than buffer + 1
            eroded_diff = binary_erosion(diff.squeeze(), np.ones((buffer + 1, buffer + 1)))
            assert np.count_nonzero(eroded_diff) == 0

        # Test that buffer < 0 works
        vector_5 = self.vector_5
        out_mask = vector_5.create_mask(rst, as_array=True)
        for buffer in np.arange(-1, -3, -1):
            out_mask_buff = vector_5.create_mask(rst, buffer=buffer, as_array=True)
            diff = ~out_mask_buff & out_mask
            assert np.count_nonzero(diff) > 0
            # Difference between masks should always be thinner than buffer + 1
            eroded_diff = binary_erosion(diff.squeeze(), np.ones((abs(buffer) + 1, abs(buffer) + 1)))
            assert np.count_nonzero(eroded_diff) == 0

        # Check that no warning is raised when creating a mask with a xres not multiple of vector bounds
        mask = vector.create_mask(xres=1.01)

        # Check that by default, create_mask returns a Mask
        assert isinstance(mask, gu.Mask)

        # Check that an error is raised if xres is not passed
        with pytest.raises(ValueError, match="At least raster or xres must be set."):
            vector.create_mask()

        # Check that an error is raised if buffer is the wrong type
        with pytest.raises(TypeError, match="Buffer must be a number, currently set to str."):
            vector.create_mask(rst, buffer="lol")  # type: ignore

        # If the raster has the wrong type
        with pytest.raises(TypeError, match="Raster must be a geoutils.Raster or None."):
            vector.create_mask("lol")  # type: ignore

        # Check that a warning is raised if the bounds were passed specifically by the user
        with pytest.warns(UserWarning):
            vector.create_mask(xres=1.01, bounds=(0, 0, 21, 21))

    def test_extract_vertices(self) -> None:
        """
        Test that extract_vertices works with simple geometries.
        """
        # Polygons
        vertices = gu.vector.extract_vertices(self.vector.ds)
        assert len(vertices) == 1
        assert vertices == [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]]

        # MultiPolygons
        vertices = gu.vector.extract_vertices(self.vector_multipoly.ds)
        assert len(vertices) == 2
        assert vertices[0] == [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]
        assert vertices[1] == [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0), (5.0, 5.0)]

        # LineString
        vertices = gu.vector.extract_vertices(self.vector_lines.ds)
        assert len(vertices) == 1
        assert vertices == [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)]]

        # MultiLineString
        vertices = gu.vector.extract_vertices(self.vector_multilines.ds)
        assert len(vertices) == 2
        assert vertices[0] == [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)]
        assert vertices[1] == [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0)]

    def test_generate_voronoi(self) -> None:
        """
        Check that vector.generate_voronoi_polygons works on a simple Polygon.
        Does not work with simple shapes as squares or triangles as the diagram is infinite.
        For now, test on a set of two squares.
        """
        # Check with a multipolygon
        voronoi = gu.vector.generate_voronoi_polygons(self.vector_multipoly.ds)
        assert len(voronoi) == 2
        vertices = gu.vector.extract_vertices(voronoi)
        assert vertices == [
            [(5.5, 10.5), (10.5, 10.5), (10.5, 5.5), (5.5, 10.5)],
            [(5.5, 10.5), (10.5, 5.5), (5.5, 5.5), (5.5, 10.5)],
        ]

        # Check that it fails with proper error for too simple geometries
        expected_message = "Invalid geometry, cannot generate finite Voronoi polygons"
        with pytest.raises(ValueError, match=expected_message):
            voronoi = gu.vector.generate_voronoi_polygons(self.vector.ds)

    def test_buffer_metric(self) -> None:
        """Check that metric buffering works"""

        # Case with two squares: test that the buffered area is without deformations
        # https://epsg.io/32631
        utm31_x_center = 500000
        utm31_y_center = 4649776
        poly1_utm31 = Polygon(
            [
                (utm31_x_center, utm31_y_center),
                (utm31_x_center + 1, utm31_y_center),
                (utm31_x_center + 1, utm31_y_center + 1),
                (utm31_x_center, utm31_y_center + 1),
            ]
        )

        poly2_utm31 = Polygon(
            [
                (utm31_x_center + 10, utm31_y_center + 10),
                (utm31_x_center + 11, utm31_y_center + 10),
                (utm31_x_center + 11, utm31_y_center + 11),
                (utm31_x_center + 10, utm31_y_center + 11),
            ]
        )

        # We initiate the squares of size 1x1 in a UTM projection
        two_squares = gu.Vector(gpd.GeoDataFrame(geometry=[poly1_utm31, poly2_utm31], crs="EPSG:32631"))

        # Their area should now be 1 for each polygon
        assert two_squares.ds.area.values[0] == 1
        assert two_squares.ds.area.values[1] == 1

        # We buffer them
        two_squares_utm_buffered = two_squares.buffer_metric(buffer_size=1.0)

        # Their area should now be 1 (square) + 4 (buffer along the sides) + 4*(pi*1**2 /4)
        # (buffer of corners = quarter-disks)
        expected_area = 1 + 4 + np.pi
        assert two_squares_utm_buffered.ds.area.values[0] == pytest.approx(expected_area, abs=0.01)
        assert two_squares_utm_buffered.ds.area.values[1] == pytest.approx(expected_area, abs=0.01)

        # And the new GeoDataFrame should exactly match that of one buffer from the original one
        direct_gpd_buffer = gu.Vector(
            gpd.GeoDataFrame(geometry=two_squares.ds.buffer(distance=1.0).geometry, crs=two_squares.crs)
        )
        assert_geodataframe_equal(direct_gpd_buffer.ds, two_squares_utm_buffered.ds)

        # Now, if we reproject the original vector in a non-metric system
        two_squares_geographic = gu.Vector(two_squares.ds.to_crs(epsg=4326))
        # We buffer directly the Vector object in the non-metric system
        two_squares_geographic_buffered = two_squares_geographic.buffer_metric(buffer_size=1.0)
        # Then, we reproject that vector in the UTM zone
        two_squares_geographic_buffered_reproj = gu.Vector(
            two_squares_geographic_buffered.ds.to_crs(crs=two_squares.crs)
        )

        # Their area should now be the same as before for each polygon
        assert two_squares_geographic_buffered_reproj.ds.area.values[0] == pytest.approx(expected_area, abs=0.01)
        assert two_squares_geographic_buffered_reproj.ds.area.values[0] == pytest.approx(expected_area, abs=0.01)

        # And this time, it is the reprojected GeoDataFrame that should almost match (within a tolerance of 10e-06)
        assert all(direct_gpd_buffer.ds.geom_equals_exact(two_squares_geographic_buffered_reproj.ds, tolerance=10e-6))

    def test_buffer_without_overlap(self, monkeypatch) -> None:  # type: ignore
        """
        Check that non-overlapping buffer feature works. Does not work on simple geometries, so test on MultiPolygon.
        Yet, very simple geometries yield unexpected results, as is the case for the second test case here.
        """
        # Case 1, test with two squares, in separate Polygons
        two_squares = gu.Vector(gpd.GeoDataFrame(geometry=[self.poly1, self.poly2], crs="EPSG:4326"))

        # Check with buffers that should not overlap
        # ------------------------------------------
        buffer_size = 2
        # We force metric = False, so buffer should raise a GeoPandas warning
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS.*"):
            buffer = two_squares.buffer_without_overlap(buffer_size, metric=False)

        # Output should be of same size as input and same geometry type
        assert len(buffer.ds) == len(two_squares.ds)
        assert np.all(buffer.ds.geometry.geom_type == two_squares.ds.geometry.geom_type)

        # Extract individual geometries
        polys = []
        for geom in buffer.ds.geometry:
            if geom.geom_type in ["MultiPolygon"]:
                polys.extend(list(geom))
            else:
                polys.append(geom)

        # Check they do not overlap
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert not polys[i].intersects(polys[j])

        # buffer should yield the same result as create_mask with buffer, minus the original mask
        mask_nonoverlap = buffer.create_mask(xres=0.1, bounds=(0, 0, 21, 21))
        mask_buffer = two_squares.create_mask(xres=0.1, bounds=(0, 0, 21, 21), buffer=buffer_size)
        mask_nobuffer = two_squares.create_mask(xres=0.1, bounds=(0, 0, 21, 21))
        assert np.all(mask_nobuffer | mask_nonoverlap == mask_buffer)

        # Case 2 - Check with buffers that overlap -> this case is actually not the expected result !
        # -------------------------------
        buffer_size = 5
        # We force metric = False, so buffer should raise a GeoPandas warning
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS.*"):
            buffer = two_squares.buffer_without_overlap(buffer_size, metric=False)

        # Output should be of same size as input and same geometry type
        assert len(buffer.ds) == len(two_squares.ds)
        assert np.all(buffer.ds.geometry.geom_type == two_squares.ds.geometry.geom_type)

        # Extract individual geometries
        polys = []
        for geom in buffer.ds.geometry:
            if geom.geom_type in ["MultiPolygon"]:
                polys.extend(list(geom))
            else:
                polys.append(geom)

        # Check they do not overlap
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                assert polys[i].intersection(polys[j]).area == 0

        # buffer should yield the same result as create_mask with buffer, minus the original mask
        mask_nonoverlap = buffer.create_mask(xres=0.1, bounds=(0, 0, 21, 21))
        mask_buffer = two_squares.create_mask(xres=0.1, bounds=(0, 0, 21, 21), buffer=buffer_size)
        mask_nobuffer = two_squares.create_mask(xres=0.1, bounds=(0, 0, 21, 21))
        assert np.all(mask_nobuffer | mask_nonoverlap == mask_buffer)

        # Check that plotting runs without errors and close it
        monkeypatch.setattr(plt, "show", lambda: None)
        two_squares.buffer_without_overlap(buffer_size, plot=True)


class NeedToImplementWarning(FutureWarning):
    """Warning to remember to implement new GeoPandas methods"""


class TestGeoPandasMethods:
    # Use two synthetic vectors
    poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    gdf1 = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    synthvec1 = gu.Vector(gdf1)

    # Create a synthetic LineString geometry
    lines = LineString([(10, 10), (10.5, 10.5), (11, 11)])
    gdf2 = gpd.GeoDataFrame({"geometry": [lines]}, crs="EPSG:4326")
    synthvec2 = gu.Vector(gdf2)

    # Use two real-life vectors
    realvec1 = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))
    realvec2 = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))

    # Properties and methods derived from Shapely or GeoPandas
    # List of properties and methods with non-geometric output that are implemented in GeoUtils
    main_properties = ["crs", "geometry", "total_bounds"]
    nongeo_properties = [
        "area",
        "length",
        "interiors",
        "geom_type",
        "is_empty",
        "is_ring",
        "is_simple",
        "is_valid",
        "has_z",
    ]
    nongeo_methods = [
        "contains",
        "geom_equals",
        "geom_almost_equals",
        "geom_equals_exact",
        "crosses",
        "disjoint",
        "intersects",
        "overlaps",
        "touches",
        "within",
        "covers",
        "covered_by",
        "distance",
    ]

    # List of properties and methods with geometric output that are implemented in GeoUtils
    geo_properties = ["boundary", "unary_union", "centroid", "convex_hull", "envelope", "exterior"]
    geo_methods = [
        "representative_point",
        "normalize",
        "make_valid",
        "difference",
        "symmetric_difference",
        "union",
        "intersection",
        "clip_by_rect",
        "buffer",
        "simplify",
        "affine_transform",
        "translate",
        "rotate",
        "scale",
        "skew",
        "dissolve",
        "explode",
        "sjoin",
        "sjoin_nearest",
        "overlay",
        "to_crs",
        "set_crs",
        "rename_geometry",
        "set_geometry",
        "clip",
    ]
    # List of class methods
    io_methods = [
        "from_file",
        "from_postgis",
        "from_dict",
        "from_features",
        "to_feather",
        "to_parquet",
        "to_file",
        "to_postgis",
        "to_json",
        "to_wkb",
        "to_wkt",
        "to_csv",
    ]

    # List of other properties and methods
    other = ["has_sindex", "sindex", "estimate_utm_crs", "cx", "iterfeatures"]
    all_declared = (
        main_properties + nongeo_methods + nongeo_properties + geo_methods + geo_properties + other + io_methods
    )

    # Exceptions for GeoPandasBase functions not implemented (or deprecrated) in GeoSeries/GeoDataFrame
    exceptions_unimplemented = [
        "plot",
        "explore",
        "cascaded_union",
        "bounds",
        "relate",
        "project",
        "interpolate",
        "equals",
        "type",
        "convert_dtypes",
        "merge",
        "apply",
        "astype",
        "minimum_bounding_circle",
        "minimum_bounding_radius",
        "get_coordinates",
        "hilbert_distance",
        "sample_points",
        "copy",
    ]
    # Exceptions for IO/conversion that can be done directly from .ds
    all_exceptions = exceptions_unimplemented

    # Get all GeoPandasBase public methods with some exceptions
    geobase_methods = gpd.base.GeoPandasBase.__dict__.copy()

    # Get all GeoDataFrame public methods with some exceptions
    gdf_methods = gpd.GeoDataFrame.__dict__.copy()

    def test_overridden_funcs_exist(self) -> None:
        """Check that all methods listed above exist in Vector."""

        # Check that all methods declared in the class above exist in Vector
        vector_methods = gu.Vector.__dict__

        list_missing = [method for method in self.all_declared if method not in vector_methods.keys()]

        assert len(list_missing) == 0, print(f"Test method listed that is not in GeoUtils: {list_missing}")

    def test_geopandas_coverage(self) -> None:
        """Check that all existing methods of GeoPandas are overridden, with a couple exceptions."""

        # Merge the two
        all_methods = self.geobase_methods.copy()
        all_methods.update(self.gdf_methods)

        # Remove exceptions we don't want to reuse from GeoPandas (mirrored in Vector)
        name_all_methods = list(all_methods.keys())
        public_methods = [method for method in name_all_methods if method[0] != "_"]

        covered_methods = [method for method in public_methods if method not in self.all_exceptions]

        # Check that all methods declared in the class above are covered in Vector
        list_missing = [method for method in covered_methods if method not in self.all_declared]

        if len(list_missing) != 0:
            warnings.warn(
                f"New GeoPandas methods are not implemented in GeoUtils: {list_missing}", NeedToImplementWarning
            )

    @pytest.mark.parametrize("method", nongeo_methods + geo_methods)  # type: ignore
    def test_overridden_funcs_args(self, method: str) -> None:
        """Check that all methods overridden have the same arguments as in GeoPandas."""

        # Get GeoPandas class where the methods live
        if method in self.geobase_methods.keys():
            upstream_class = gpd.base.GeoPandasBase
        elif method in self.gdf_methods.keys():
            upstream_class = gpd.GeoDataFrame
        else:
            raise ValueError("Method did not belong to GeoDataFrame or GeoPandasBase class.")

        # Get a full argument inspection object for each class
        argspec_upstream = inspect.getfullargspec(getattr(upstream_class, method))
        argspec_geoutils = inspect.getfullargspec(getattr(gu.Vector, method))

        # Check that all positional arguments are the same
        if argspec_upstream.args != argspec_geoutils.args:
            warnings.warn("Argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

        # Check that the *args and **kwargs argument are declared consistently
        if argspec_upstream.varargs != argspec_geoutils.varargs:
            warnings.warn("Argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

        if argspec_upstream.varkw != argspec_geoutils.varkw:
            warnings.warn("Argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

        # Check that default argument values are the same
        if argspec_upstream.defaults != argspec_geoutils.defaults:
            warnings.warn("Default argument of GeoPandas method not consistent in GeoUtils.", NeedToImplementWarning)

    @pytest.mark.parametrize("vector", [synthvec1, synthvec2, realvec1, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", nongeo_properties)  # type: ignore
    def test_nongeo_properties(self, vector: gu.Vector, method: str) -> None:
        """Check non-geometric properties are consistent with GeoPandas."""

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        output_geoutils = getattr(vector, method)
        output_geopandas = getattr(vector.ds, method)

        # Assert equality
        assert_series_equal(output_geoutils, output_geopandas)

    @pytest.mark.parametrize("vector1", [synthvec1, realvec1])  # type: ignore
    @pytest.mark.parametrize("vector2", [synthvec2, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", nongeo_methods)  # type: ignore
    def test_nongeo_methods(self, vector1: gu.Vector, vector2: gu.Vector, method: str) -> None:
        """
        Check non-geometric methods are consistent with GeoPandas.
        All these methods require two inputs ("other", "df", or "right" argument), except one.
        """

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        if method != "geom_equals_exact":
            output_geoutils = getattr(vector1, method)(vector2)
            output_geopandas = getattr(vector1.ds, method)(vector2.ds)
        else:
            output_geoutils = getattr(vector1, method)(vector2, tolerance=0.1)
            output_geopandas = getattr(vector1.ds, method)(vector2.ds, tolerance=0.1)

        # Assert equality
        assert_series_equal(output_geoutils, output_geopandas)

    @pytest.mark.parametrize("vector", [synthvec1, synthvec2, realvec1, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", geo_properties)  # type: ignore
    def test_geo_properties(self, vector: gu.Vector, method: str) -> None:
        """Check geometric properties are consistent with GeoPandas."""

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        output_geoutils = getattr(vector, method)
        output_geopandas = getattr(vector.ds, method)

        # Assert output types
        assert isinstance(output_geoutils, gu.Vector)
        assert isinstance(output_geopandas, (gpd.GeoSeries, gpd.GeoDataFrame, BaseGeometry))

        # Separate cases depending on GeoPandas' output
        if isinstance(output_geopandas, gpd.GeoSeries):
            # Assert geoseries equality
            assert_geoseries_equal(output_geoutils.ds.geometry, output_geopandas)
        elif isinstance(output_geopandas, BaseGeometry):
            assert_geodataframe_equal(
                output_geoutils.ds, gpd.GeoDataFrame({"geometry": [output_geopandas]}, crs=vector.crs)
            )
        else:
            assert_geodataframe_equal(output_geoutils.ds, output_geopandas)

    specific_method_args = {
        "buffer": {"distance": 1},
        "clip_by_rect": {"xmin": 10.5, "ymin": 10.5, "xmax": 11, "ymax": 11},
        "affine_transform": {"matrix": [1, 1, 1, 1, 1, 1]},
        "translate": {"xoff": 1, "yoff": 1, "zoff": 0},
        "rotate": {"angle": 90},
        "scale": {"xfact": 1.1, "yfact": 1.1, "zfact": 1.1, "origin": "center"},
        "skew": {"xs": 1.1, "ys": 1.1},
        "interpolate": {"distance": 1},
        "simplify": {"tolerance": 0.1},
        "to_crs": {"crs": pyproj.CRS.from_epsg(32610)},
        "set_crs": {"crs": pyproj.CRS.from_epsg(32610), "allow_override": True},
        "rename_geometry": {"col": "lol"},
        "set_geometry": {"col": synthvec1.geometry},
        "clip": {"mask": poly},
    }

    @pytest.mark.parametrize("vector1", [synthvec1, realvec1])  # type: ignore
    @pytest.mark.parametrize("vector2", [synthvec2, realvec2])  # type: ignore
    @pytest.mark.parametrize("method", geo_methods)  # type: ignore
    def test_geo_methods(self, vector1: gu.Vector, vector2: gu.Vector, method: str) -> None:
        """Check geometric methods are consistent with GeoPandas."""

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Methods that require two inputs
        if method in [
            "difference",
            "symmetric_difference",
            "union",
            "intersection",
            "sjoin",
            "sjoin_nearest",
            "overlay",
        ]:
            output_geoutils = getattr(vector1, method)(vector2)
            output_geopandas = getattr(vector1.ds, method)(vector2.ds)
        # Methods that require zero input
        elif method in ["representative_point", "normalize", "make_valid", "dissolve", "explode"]:
            output_geoutils = getattr(vector1, method)()
            output_geopandas = getattr(vector1.ds, method)()
        elif method in self.specific_method_args.keys():
            output_geoutils = getattr(vector1, method)(**self.specific_method_args[method])
            output_geopandas = getattr(vector1.ds, method)(**self.specific_method_args[method])
        else:
            raise ValueError(f"The method '{method}' is not covered by this test.")

        # Assert output types
        assert isinstance(output_geoutils, gu.Vector)
        assert isinstance(output_geopandas, (gpd.GeoSeries, gpd.GeoDataFrame))

        # Separate cases depending on GeoPandas' output, and nature of the function
        # Simplify is a special case that can make geometries invalid, so adjust test
        if method == "simplify":
            # TODO: Unskip this random test failure (one index not matching) when this is fixed in GeoPandas/Shapely
            pass
            # assert_geoseries_equal(
            #     output_geopandas.make_valid(), output_geoutils.ds.geometry.make_valid(), check_less_precise=True
            # )
        # For geoseries output, check equality of it
        elif isinstance(output_geopandas, gpd.GeoSeries):
            assert_geoseries_equal(output_geoutils.ds.geometry, output_geopandas)
        # For geodataframe output, check equality
        else:
            assert_geodataframe_equal(output_geoutils.ds, output_geopandas)
