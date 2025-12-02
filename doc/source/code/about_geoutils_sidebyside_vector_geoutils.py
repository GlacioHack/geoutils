####
# Part not shown in the doc, to get data files ready
import geoutils

landsat_b4_path = geoutils.examples.get_path("everest_landsat_b4")
everest_outlines_path = geoutils.examples.get_path("everest_rgi_outlines")
geoutils.Raster(landsat_b4_path).to_file("myraster.tif")
geoutils.Vector(everest_outlines_path).to_file("myvector.gpkg")
####

import geoutils as gu

# Opening a vector and a raster
vect = gu.Vector("myvector.gpkg")
rast = gu.Raster("myraster.tif")

# Metric buffering
vect_buff = vect.buffer_metric(buffer_size=100)

# Create a mask on the raster grid
# (raster not loaded, only metadata)
mask = vect_buff.create_mask(rast)

# Index raster values on mask
# (raster loads implicitly)
values = rast[mask]

####
# Part not shown in the doc, to get data files ready
import os

for file in ["myraster.tif", "myvector.gpkg"]:
    os.remove(file)
####
