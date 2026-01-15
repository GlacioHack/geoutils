####
# Part not shown in the doc, to get data files ready
import geoutils

landsat_b4_path = geoutils.examples.get_path("everest_landsat_b4")
landsat_b4_crop_path = geoutils.examples.get_path("everest_landsat_b4_cropped")
geoutils.Raster(landsat_b4_path).to_file("myraster1.tif")
geoutils.Raster(landsat_b4_crop_path).to_file("myraster2.tif")
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="For reprojection, nodata must be set.*")
warnings.filterwarnings("ignore", category=UserWarning, message="No nodata set*")
warnings.filterwarnings("ignore", category=UserWarning, message="One raster has a pixel interpretation*")
####

import geoutils as gu

# Opening of two rasters
rast1 = gu.Raster("myraster1.tif")
rast2 = gu.Raster("myraster2.tif")

# Reproject 1 to match 2
# (raster 2 not loaded, only metadata)
rast1_reproj = rast1.reproject(ref=rast2)

# Array interfacing and implicit loading
# (raster 2 loads implicitly)
rast_result = (1 + rast2) / rast1_reproj

# Saving
rast_result.to_file("myresult.tif")

####
# Part not shown in the docs, to clean up
import os

for file in ["myraster1.tif", "myraster2.tif", "myresult.tif"]:
    os.remove(file)
####
