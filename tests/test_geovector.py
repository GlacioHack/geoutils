import geoutils as gu

GLACIER_OUTLINES_URL = "http://public.data.npolar.no/cryoclim/CryoClim_GAO_SJ_1990.zip"


class TestVector:
    glacier_outlines = gu.Vector(GLACIER_OUTLINES_URL)

    def test_init(self):

        vector = gu.Vector(GLACIER_OUTLINES_URL)

        assert isinstance(vector, gu.Vector)

    def test_copy(self):

        vector2 = self.glacier_outlines.copy()

        assert vector2 is not self.glacier_outlines

        vector2.ds = vector2.ds.query("NAME == 'Ayerbreen'")

        assert vector2.ds.shape[0] < self.glacier_outlines.ds.shape[0]
