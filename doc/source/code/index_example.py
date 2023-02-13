import geoutils as gu

# Examples files
filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")

# Open files
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)

# Crop raster to vector's extent
rast.crop(vect)

# Proximity to vector on raster's grid
rast_proximity_to_vec = vect.proximity(rast)

# Add 1 to the raster array
rast += 1

# Apply a sine to the raster array
import numpy as np
rast = np.sin(rast)

# Get mask where proximity to vector fits specific raster values
mask_aoi = rast >= rast_proximity_to_vec

# Index raster with the mask to extract a 1-D array of values
values_aoi = rast[mask_aoi]

# Polygonize areas where mask is True
vect_aoi = mask_aoi.polygonize()

# Read file as a satellite image
print(filename_rast)
satimg = gu.SatelliteImage(filename_rast, silent=False)

