import geoutils as gu

# Examples files
filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")

# Open files
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)

# Print vector metadata info
print(vect.info())