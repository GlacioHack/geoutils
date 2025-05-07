"""Tests for geometry operations on vectors."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal
from shapely import LineString, MultiLineString, MultiPolygon, Polygon

import geoutils as gu
from geoutils.vector.geometric import _extract_vertices, _generate_voronoi_polygons


class TestGeometric:

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

    def test_extract_vertices(self) -> None:
        """
        Test that extract_vertices works with simple geometries.
        """
        # Polygons
        vertices = _extract_vertices(self.vector.ds)
        assert len(vertices) == 1
        assert vertices == [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]]

        # MultiPolygons
        vertices = _extract_vertices(self.vector_multipoly.ds)
        assert len(vertices) == 2
        assert vertices[0] == [(10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0), (10.0, 10.0)]
        assert vertices[1] == [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0), (5.0, 5.0)]

        # LineString
        vertices = _extract_vertices(self.vector_lines.ds)
        assert len(vertices) == 1
        assert vertices == [[(10.0, 10.0), (11.0, 10.0), (11.0, 11.0)]]

        # MultiLineString
        vertices = _extract_vertices(self.vector_multilines.ds)
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
        voronoi = _generate_voronoi_polygons(self.vector_multipoly.ds)
        assert len(voronoi) == 2
        vertices = _extract_vertices(voronoi)
        assert vertices == [
            [(5.5, 10.5), (10.5, 10.5), (10.5, 5.5), (5.5, 10.5)],
            [(5.5, 10.5), (10.5, 5.5), (5.5, 5.5), (5.5, 10.5)],
        ]

        # Check that it fails with proper error for too simple geometries
        expected_message = "Invalid geometry, cannot generate finite Voronoi polygons"
        with pytest.raises(ValueError, match=expected_message):
            voronoi = _generate_voronoi_polygons(self.vector.ds)

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
        assert buffer.crs == two_squares.crs

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
        mask_nonoverlap = buffer.create_mask(res=0.1, bounds=(0, 0, 21, 21))
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS.*"):
            two_squares_buffer = two_squares.buffer(buffer_size)
        mask_buffer = two_squares_buffer.create_mask(res=0.1, bounds=(0, 0, 21, 21))
        mask_nobuffer = two_squares.create_mask(res=0.1, bounds=(0, 0, 21, 21))
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
        mask_nonoverlap = buffer.create_mask(res=0.1, bounds=(0, 0, 21, 21))
        with pytest.warns(UserWarning, match="Geometry is in a geographic CRS.*"):
            two_squares_buffer = two_squares.buffer(buffer_size)
        mask_buffer = two_squares_buffer.create_mask(res=0.1, bounds=(0, 0, 21, 21))
        mask_nobuffer = two_squares.create_mask(res=0.1, bounds=(0, 0, 21, 21))
        assert np.all(mask_nobuffer | mask_nonoverlap == mask_buffer)

        # Check that plotting runs without errors and close it
        monkeypatch.setattr(plt, "show", lambda: None)
        two_squares.buffer_without_overlap(buffer_size, plot=True)
