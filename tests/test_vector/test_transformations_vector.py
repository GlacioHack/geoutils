"""Tests for geotransformations of vectors."""

from __future__ import annotations

import re
from typing import Any

import geopandas as gpd
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from shapely.geometry import Polygon, box

import geoutils as gu
from geoutils.exceptions import InvalidBoundsError
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster


class TestTransformation:
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
        with pytest.raises(TypeError, match="Match-reference input must have a 'crs' attribute.*"):
            v0.reproject(ref=10)  # type: ignore

    test_data = [[landsat_b4_path, everest_outlines_path], [aster_dem_path, aster_outlines_path]]

    @pytest.mark.parametrize("data", test_data)
    def test_crop(self, data: list[str]) -> None:
        """Checks that crop() keeps unchanged geometries that intersect a bounding box."""

        # Load data
        raster_path, outlines_path = data
        rst = gu.Raster(raster_path)
        outlines = gu.Vector(outlines_path)

        # Need to reproject to r.crs. Otherwise, crop will work but will be approximate
        # Because outlines might be warped in a different crs
        outlines.ds = outlines.ds.to_crs(rst.crs)

        # Crop
        outlines_new = outlines.copy()
        outlines_new.crop(rst, inplace=True)

        # Check default behaviour - crop and return copy
        outlines_copy = outlines.crop(rst)

        # Crop by passing bounds
        outlines_new_bounds = outlines.copy()
        outlines_new_bounds.crop(list(rst.bounds), inplace=True)
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
        with pytest.raises(InvalidBoundsError, match="Cannot interpret bounding box input.*"):
            outlines.crop(1, inplace=True)  # type: ignore

    def test_crop__mode(self) -> None:
        """Checks that crop() can keep intersecting or fully contained geometries."""

        # Create one polygon inside the box, one crossing its edge and one outside it
        bbox = (0, 0, 3, 3)
        source = gpd.GeoDataFrame(
            {"name": ["inside", "crossing", "outside"]},
            geometry=[box(1, 1, 2, 2), box(-1, 1, 1, 2), box(5, 5, 6, 6)],
            crs=32610,
        )
        vector = gu.Vector(source)

        # Check that the default mode keeps both polygons that touch the box
        intersecting = vector.crop(bbox)
        assert list(intersecting["name"]) == ["inside", "crossing"]
        assert_geodataframe_equal(intersecting.ds, source.iloc[:2])

        # Check that the within mode keeps only the polygon inside the box
        contained = vector.crop(bbox, mode="within")
        assert list(contained["name"]) == ["inside"]
        assert_geodataframe_equal(contained.ds, source.iloc[[0]])

        # Reject an unknown selection mode
        with pytest.raises(ValueError, match="must be either 'intersects' or 'within'"):
            vector.crop(bbox, mode="overlaps")  # type: ignore[arg-type]

    def test_crop__deferred(self, tmp_path: Any) -> None:
        """Checks that crop() does not read geometries from a file until they are needed."""

        # Write one polygon inside the box, one crossing its edge and one outside it
        source = gpd.GeoDataFrame(
            {"name": ["inside", "crossing", "outside"]},
            geometry=[box(1, 1, 2, 2), box(-1, 1, 1, 2), box(5, 5, 6, 6)],
            crs=32610,
        )
        path = tmp_path / "polygons.gpkg"
        source.to_file(path)
        vector = gu.Vector(path)

        # Select the polygon inside the box and keep both objects unloaded
        cropped = vector.crop((0, 0, 3, 3), mode="within")
        assert not vector.is_loaded
        assert not cropped.is_loaded

        # Read the result and compare it with the same crop in memory
        expected = gu.Vector(source).crop((0, 0, 3, 3), mode="within")
        assert cropped.vector_equal(expected)
        assert not vector.is_loaded

    def test_clip(self) -> None:
        """Checks that clip() cuts polygons at the edge of the given shape."""

        # Create a polygon crossing the clip boundary and another polygon outside it
        source = gpd.GeoDataFrame(
            {"name": ["crossing", "outside"]},
            geometry=[box(-1, 1, 2, 2), box(5, 5, 6, 6)],
            crs=32610,
        )
        clipping_geometry = box(0, 0, 1, 3)

        # Compare the clipped polygon with GeoPandas and check its new bounds
        clipped = gu.Vector(source).clip(clipping_geometry)
        expected = source.clip(clipping_geometry)
        assert_geodataframe_equal(clipped.ds, expected)
        assert tuple(clipped.bounds) == (0, 1, 1, 2)

        # Check that the old crop option warns and keeps its previous behavior
        with pytest.warns(DeprecationWarning, match="Argument 'clip' is deprecated"):
            deprecated_clipped = gu.Vector(source).crop(clipping_geometry.bounds, clip=True)
        assert_geodataframe_equal(deprecated_clipped.ds, expected)
        with pytest.warns(DeprecationWarning, match="Argument 'clip' is deprecated"):
            deprecated_cropped = gu.Vector(source).crop(clipping_geometry.bounds, clip=False)
        assert_geodataframe_equal(deprecated_cropped.ds, gu.Vector(source).crop(clipping_geometry.bounds).ds)
        inplace = gu.Vector(source)
        with pytest.warns(DeprecationWarning, match="Argument 'clip' is deprecated"):
            output = inplace.crop(clipping_geometry.bounds, clip=True, inplace=True)
        assert output is None
        assert_geodataframe_equal(inplace.ds, expected)

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


class TestTransformationChunked:
    """Test module for vector transformations run with Dask or multiprocessing."""

    def test_clip__chunked_backends_equal(self, tmp_path: Any) -> None:
        """Checks that clip with Dask and multiprocessing gives the same result as in-memory."""

        dgpd = pytest.importorskip("dask_geopandas")

        # Write five polygons with one outside and one that will be cut
        source = gpd.GeoDataFrame(
            {"row_id": np.arange(5, dtype=np.int32), "value": np.linspace(0, 1, 5)},
            geometry=[
                box(7, 7, 8, 8),
                box(-4, 1, -2, 2),
                box(-1, 0, 1, 2),
                box(1, 0, 3, 2),
                box(3, 0, 5, 2),
            ],
            crs=32610,
        )
        filename = tmp_path / "vector_source.gpkg"
        source.to_file(filename, index=False)
        geometry = Polygon([(0, 0), (6, 0), (0, 6)])

        # Clip the same file in memory and with Dask + MP
        expected = source.clip(geometry).sort_values("row_id").reset_index(drop=True)
        lazy = gu.open_vector(filename, chunks=2)
        multiproc = gu.Vector(filename)
        lazy_result = lazy.vct.clip(geometry)
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=2, outfile=str(tmp_path / "vector_clipped.gpkg"), cluster=cluster)
            multiproc_result = multiproc.clip(geometry, mp_config=config)

        # Check that the Dask and MP outputs have expected types and remain unloaded
        assert isinstance(lazy_result, dgpd.GeoDataFrame)
        assert not lazy.vct.is_loaded and not lazy_result.vct.is_loaded
        assert isinstance(multiproc_result, gu.Vector)
        assert not multiproc.is_loaded and not multiproc_result.is_loaded
        with pytest.raises(ValueError, match="cannot be combined with a Dask vector"):
            lazy.vct.clip(geometry, mp_config=config)

        # Read results and check exact equality of clipped geometry in original order with in-memory
        computed_lazy = lazy_result.compute().sort_values("row_id").reset_index(drop=True)
        computed_multiproc = multiproc_result.ds.sort_values("row_id").reset_index(drop=True)
        assert_geodataframe_equal(computed_lazy, expected, check_dtype=False)
        assert_geodataframe_equal(computed_multiproc, expected, check_dtype=False)
        assert not multiproc.is_loaded
        assert not lazy.vct.is_loaded and not lazy_result.vct.is_loaded
