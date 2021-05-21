import geoutils as gu

filename = gu.datasets.get_path("landsat_B4")

raster = gu.Raster(filename)

print(raster)
