from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from scipy.ndimage import binary_erosion
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

        vector = gu.Vector(GLACIER_OUTLINES_URL)

        assert isinstance(vector, gu.Vector)

    def test_copy(self) -> None:

        vector2 = self.glacier_outlines.copy()

        assert vector2 is not self.glacier_outlines

        vector2.ds = vector2.ds.query("NAME == 'Ayerbreen'")

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]

    def test_query(self) -> None:

        vector2 = self.glacier_outlines.query("NAME == 'Ayerbreen'")

        assert vector2 is not self.glacier_outlines

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]

    def test_bounds(self) -> None:

        bounds = self.glacier_outlines.bounds

        assert bounds.left < bounds.right
        assert bounds.bottom < bounds.top

        assert bounds.left == self.glacier_outlines.ds.total_bounds[0]
        assert bounds.bottom == self.glacier_outlines.ds.total_bounds[1]
        assert bounds.right == self.glacier_outlines.ds.total_bounds[2]
        assert bounds.top == self.glacier_outlines.ds.total_bounds[3]

    def test_rasterize_proj(self) -> None:

        # Capture the warning on resolution not matching exactly bounds
        with pytest.warns(UserWarning):
            burned = self.glacier_outlines.rasterize(xres=3000)

        assert burned.shape[0] == 146
        assert burned.shape[1] == 115

    def test_rasterize_unproj(self) -> None:
        """Test rasterizing an EPSG:3426 dataset into a projection."""
        v = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))
        # Use Web Mercator at 30 m.

        # Capture the warning on resolution not matching exactly bounds
        with pytest.warns(UserWarning):
            burned = v.rasterize(xres=30, crs=3857)

        assert burned.shape[0] == 1251
        assert burned.shape[1] == 1522

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
        outlines_new.crop(rst)

        # Check with bracket call
        outlines_new2 = outlines_new[rst]
        assert_geodataframe_equal(outlines_new.ds, outlines_new2.ds)

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

    def test_proximity(self) -> None:
        """
        The core functionality is already tested against GDAL in test_raster: just verify the vector-specific behaviour.
        #TODO: add an artificial test as well (mirroring TODO in test_georaster)
        """

        vector = gu.Vector(self.everest_outlines_path)

        # -- Test 1: with a Raster provided --
        raster1 = gu.Raster(self.landsat_b4_crop_path)
        prox1 = vector.proximity(raster=raster1)

        # The proximity should have the same extent, resolution and CRS
        assert raster1.equal_georeferenced_grid(prox1)

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
        vector.proximity(grid_size=(100, 100))


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
        out_mask = vector.create_mask(xres=1, bounds=(0, 0, 21, 21))
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
        out_mask = vector.create_mask(rst)
        assert out_mask.shape == (1, 21, 21)

        # With gu.Raster, 2 bands -> fails...
        # rst = gu.Raster.from_array(np.zeros((2, 21, 21)), transform=(1., 0., 0., 0., -1., 21.), crs='EPSG:4326')
        # out_mask = vector.create_mask(rst)

        # Test that buffer = 0 works
        out_mask_buff = vector.create_mask(rst, buffer=0)
        assert np.all(ref_mask == out_mask_buff)

        # Test that buffer > 0 works
        rst = gu.Raster.from_array(np.zeros((21, 21)), transform=(1.0, 0.0, 0.0, 0.0, -1.0, 21.0), crs="EPSG:4326")
        out_mask = vector.create_mask(rst)
        for buffer in np.arange(1, 8):
            out_mask_buff = vector.create_mask(rst, buffer=buffer)
            diff = out_mask_buff & ~out_mask
            assert np.count_nonzero(diff) > 0
            # Difference between masks should always be thinner than buffer + 1
            eroded_diff = binary_erosion(diff.squeeze(), np.ones((buffer + 1, buffer + 1)))
            assert np.count_nonzero(eroded_diff) == 0

        # Test that buffer < 0 works
        vector_5 = self.vector_5
        out_mask = vector_5.create_mask(rst)
        for buffer in np.arange(-1, -3, -1):
            out_mask_buff = vector_5.create_mask(rst, buffer=buffer)
            diff = ~out_mask_buff & out_mask
            assert np.count_nonzero(diff) > 0
            # Difference between masks should always be thinner than buffer + 1
            eroded_diff = binary_erosion(diff.squeeze(), np.ones((abs(buffer) + 1, abs(buffer) + 1)))
            assert np.count_nonzero(eroded_diff) == 0

        # Check that no warning is raised when creating a mask with a xres not multiple of vector bounds
        vector.create_mask(xres=1.01)

        # Check that a warning is raised if the bounds were passed specifically by the user
        with pytest.warns(UserWarning):
            vector.create_mask(xres=1.01, bounds=(0, 0, 21, 21))

    def test_extract_vertices(self) -> None:
        """
        Test that extract_vertices works with simple geometries.
        """
        # Polygons
        vertices = gu.geovector.extract_vertices(self.vector.ds)
        assert len(vertices) == 1
        assert vertices == [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]]

        # MultiPolygons
        vertices = gu.geovector.extract_vertices(self.vector_multipoly.ds)
        assert len(vertices) == 2
        assert vertices[0] == [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]
        assert vertices[1] == [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0), (5.0, 5.0)]

        # LineString
        vertices = gu.geovector.extract_vertices(self.vector_lines.ds)
        assert len(vertices) == 1
        assert vertices == [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)]]

        # MultiLineString
        vertices = gu.geovector.extract_vertices(self.vector_multilines.ds)
        assert len(vertices) == 2
        assert vertices[0] == [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)]
        assert vertices[1] == [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0)]

    def test_generate_voronoi(self) -> None:
        """
        Check that geovector.generate_voronoi_polygons works on a simple Polygon.
        Does not work with simple shapes as squares or triangles as the diagram is infinite.
        For now, test on a set of two squares.
        """
        # Check with a multipolygon
        voronoi = gu.geovector.generate_voronoi_polygons(self.vector_multipoly.ds)
        assert len(voronoi) == 2
        vertices = gu.geovector.extract_vertices(voronoi)
        assert vertices == [
            [(5.5, 10.5), (10.5, 10.5), (10.5, 5.5), (5.5, 10.5)],
            [(5.5, 10.5), (10.5, 5.5), (5.5, 5.5), (5.5, 10.5)],
        ]

        # Check that it fails with proper error for too simple geometries
        expected_message = "Invalid geometry, cannot generate finite Voronoi polygons"
        with pytest.raises(ValueError, match=expected_message):
            voronoi = gu.geovector.generate_voronoi_polygons(self.vector.ds)

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
        assert all(direct_gpd_buffer.ds.geom_almost_equals(two_squares_geographic_buffered_reproj.ds))

    def test_buffer_without_overlap(self) -> None:
        """
        Check that non-overlapping buffer feature works. Does not work on simple geometries, so test on MultiPolygon.
        Yet, very simple geometries yield unexpected results, as is the case for the second test case here.
        """
        # Case 1, test with two squares, in separate Polygons
        two_squares = gu.Vector(gpd.GeoDataFrame(geometry=[self.poly1, self.poly2], crs="EPSG:4326"))

        # Check with buffers that should not overlap
        # ------------------------------------------
        buffer_size = 2
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
