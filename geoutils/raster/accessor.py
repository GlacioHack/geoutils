import xarray as xr

@xr.register_dataarray_accessor("rst")
class RasterAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    @property
    def crs(self):

    @property
    def transform(self):

    @property
    def nodata(self):


    @property
    def count(self):

    @property
    def height(self):

    @property
    def width(self):

    @property
    def shape(self):

    @property
    def res(self):

    @property
    def bounds(self):


    def crop(self):

    def reproject(self):

    def polygonize(self):

    def proximity(self):

    def georeferenced_grid_equal(self):

    def shift(self):

    def get_metric_crs(self):

    def get_bounds_projected(self):

    def get_footprint_projected(self):

    def subsample(self):

    def to_raster(self):

    def xy2ij(self):

    def ij2xy(self):

    def outside_image(self):

    def show(self):


@xr.register_dataarray_accessor("sat")
class SatelliteImageAccessor(RasterAccessor):
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

