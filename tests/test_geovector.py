import geopandas as gpd
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from shapely.geometry.polygon import Polygon

import geoutils as gu

GLACIER_OUTLINES_URL = "http://public.data.npolar.no/cryoclim/CryoClim_GAO_SJ_1990.zip"


class TestVector:
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


class TestSynthetic:

    # Create a synthetic vector file with a square of size 1, started at position (10, 10)
    poly = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                poly,
            ]
        },
        crs="EPSG:4326",
    )
    vector = gu.Vector(gdf)

    # Create a synthetic vector file with a square of size 5, started at position (8, 8)
    poly = Polygon([(8, 8), (13, 8), (13, 13), (8, 13)])
    gdf = gpd.GeoDataFrame(
        {
            "geometry": [
                poly,
            ]
        },
        crs="EPSG:4326",
    )
    vector_5 = gu.Vector(gdf)

    def test_create_mask(self) -> None:
        """
        Test Vector.create_mask.
        """
        # First with given res and bounds -> Should be a 21 x 21 array with 0 everywhere except center pixel
        vector = self.vector
        out_mask = vector.create_mask(xres=1, bounds=(0, 0, 21, 21))
        ref_mask = np.zeros((21, 21), dtype="bool")
        ref_mask[10, 10] = True
        assert out_mask.shape == (21, 21)
        assert np.all(ref_mask == out_mask)

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
