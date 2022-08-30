import geoutils as gu

filename = gu.examples.get_path("everest_landsat_b4")

raster = gu.Raster(filename)

print(raster)
