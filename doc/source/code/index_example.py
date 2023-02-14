import geoutils as gu

# Examples files: infrared band of Landsat and glacier outlines
filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")

# Open files
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)

# Open file as a satellite image
import os
print(os.path.basename(filename_rast))
rast = gu.SatelliteImage(filename_rast, silent=False)

# Crop raster to vector's extent
rast.crop(vect)

# Compute proximity to vector on raster's grid
rast_proximity_to_vec = vect.proximity(rast)

# Add 1 to the raster array
rast += 1

# Apply a normalization to the raster
import numpy as np
rast = (rast - np.min(rast)) / (np.max(rast) - np.min(rast))

# Get mask of an AOI: infrared index above 0.7, at least 200 m from glaciers
mask_aoi = np.logical_and(rast > 0.7, rast_proximity_to_vec > 200)

# Index raster with mask to extract a 1-D array
values_aoi = rast[mask_aoi]

# Polygonize areas where mask is True
vect_aoi = mask_aoi.polygonize()

# Plot result
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.gca()
rast.show(ax=ax, cmap='Reds', cb_title='Normalized infrared')
vect_aoi.ds.plot(ax=ax, fc='none', ec='k', lw=0.5)

# Save final raster and vector to files
# rast.save()
# vect_aoi.save()

#TODO: maybe add glacier outlines also above, once the plotting through Vector is simplified
