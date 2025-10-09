"""Tests for raster-vector interfacing."""

from __future__ import annotations

import re
import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely import LineString, MultiLineString, MultiPolygon, Polygon

import geoutils as gu
from geoutils import examples


class TestRasterVectorInterface:

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
        out_mask = vector.create_mask(res=1, bounds=(0, 0, 21, 21), as_array=True)
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

        # Check that no warning is raised when creating a mask with a xres not multiple of vector bounds
        mask = vector.create_mask(res=1.01)

        # Check that by default, create_mask returns a Mask
        assert isinstance(mask, gu.Raster) and mask.is_mask

        # Check that an error is raised if xres is not passed
        with pytest.raises(
            ValueError,
            match="Without a reference for masking, specify at least the resolution "
            "a raster mask, or the points coordinates for a point cloud mask.",
        ):
            vector.create_mask()

        # If the raster has the wrong type
        with pytest.raises(ValueError, match="Reference must be a raster or a point cloud."):
            vector.create_mask("lol")  # type: ignore

        # Check that a warning is raised if the bounds were passed specifically by the user
        with pytest.warns(UserWarning):
            vector.create_mask(res=1.01, bounds=(0, 0, 21, 21))

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_b4_crop_path = gu.examples.get_path_test("everest_landsat_b4_cropped")
    everest_outlines_path = gu.examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = gu.examples.get_path_test("exploradores_aster_dem")
    aster_outlines_path = gu.examples.get_path_test("exploradores_rgi_outlines")

    def test_rasterize(self) -> None:
        """Test rasterizing an EPSG:3426 dataset into a projection."""

        vct = gu.Vector(self.everest_outlines_path)
        rst = gu.Raster(self.landsat_b4_crop_path)

        # Use Web Mercator at 30 m.
        # Capture the warning on resolution not matching exactly bounds
        with pytest.warns(UserWarning):
            vct.rasterize(xres=30, crs=3857)

        # Typically, rasterize returns a raster
        burned_in2_out1 = vct.rasterize(raster=rst, in_value=2, out_value=1)
        assert isinstance(burned_in2_out1, gu.Raster)

        # For an in_value of 1 and out_value of 0 (default)
        burned_mask = vct.rasterize(raster=rst, in_value=1)
        assert isinstance(burned_mask, gu.Raster)
        # Convert to boolean
        burned_mask = burned_mask.astype(bool)

        # Check that rasterizing with in_value=1 is the same as creating a mask
        assert burned_mask.raster_equal(vct.create_mask(rst))

        # The two rasterization should match
        assert np.all(burned_in2_out1[burned_mask] == 2)
        assert np.all(burned_in2_out1[~burned_mask] == 1)

        # Check that errors are raised
        with pytest.raises(ValueError, match="Only one of raster or crs can be provided."):
            vct.rasterize(raster=rst, crs=3857)

    @pytest.mark.parametrize("example", [landsat_b4_path, aster_dem_path])  # type: ignore
    def test_polygonize(self, example: str) -> None:
        """Test that polygonize doesn't raise errors."""

        img = gu.Raster(example)

        # -- Test 1: basic functioning of polygonize --

        # Get unique value for image and the corresponding area
        value = np.unique(img)[0]
        pixel_area = np.count_nonzero(img.data == value) * img.res[0] * img.res[1]

        # Polygonize the raster for this value, and compute the total area
        polygonized = img.polygonize(target_values=value)
        polygon_area = polygonized.ds.area.sum()

        # Check that these two areas are approximately equal
        assert polygon_area == pytest.approx(pixel_area)
        assert isinstance(polygonized, gu.Vector)
        assert polygonized.crs == img.crs

        # Check default name of data column, and that defining a custom name works the same
        assert "id" in polygonized.ds.columns
        polygonized2 = img.polygonize(target_values=value, data_column_name="myname")
        assert "myname" in polygonized2.ds.columns
        assert np.array_equal(polygonized2.ds["myname"].values, polygonized.ds["id"].values)

        # -- Test 2: data types --

        # Check that polygonize works as expected for any input dtype (e.g. float64 being not supported by GeoPandas)
        for dtype in ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]:
            img_dtype = img.copy()
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message="dtype conversion will result in a " "loss of information.*"
                )
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Unmasked values equal to the nodata value found in data array.*",
                )
                img_dtype = img_dtype.astype(dtype)
            value = np.unique(img_dtype)[0]
            img_dtype.polygonize(target_values=value)

        # And for a boolean object, such as a mask
        mask = img > value
        mask.polygonize(target_values=1)


class TestMaskVectorInterface:

    # Paths to example data
    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    everest_outlines_path = examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    # Mask without nodata
    rst_landsat_b4 = gu.Raster(landsat_b4_path)
    mask_landsat_b4 = rst_landsat_b4 > np.nanmedian(rst_landsat_b4)
    # Mask with nodata
    rst_aster_dem = gu.Raster(aster_dem_path)
    mask_aster_dem = rst_aster_dem > np.nanmedian(rst_aster_dem)
    # Mask from an outline
    mask_everest = gu.Vector(everest_outlines_path).create_mask(gu.Raster(landsat_b4_path))

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])  # type: ignore
    def test_polygonize(self, mask: gu.Raster) -> None:
        mask_orig = mask.copy()
        # Run default
        vect = mask.polygonize()
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during polygonizing
        assert mask_orig.raster_equal(mask)

        # Check the output is cast into a vector
        assert isinstance(vect, gu.Vector)

        # Run with zero as target
        vect = mask.polygonize(target_values=0)
        assert isinstance(vect, gu.Vector)

        # Check a warning is raised when using a non-boolean value
        with pytest.warns(
            UserWarning, match=re.escape("Raster mask (boolean type) passed, using target value of 1 (" "True).")
        ):
            mask.polygonize(target_values=2)
