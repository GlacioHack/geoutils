"""
Proximity to raster or vector
=============================

This example demonstrates the calculation of proximity distances to a raster or vector using :func:`~geoutils.Raster.proximity`.
"""

# %%
# We open an example raster, and a vector for which we select a single feature

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)
vect = vect[vect["RGIId"] == "RGI60-15.10055"]
rast = rast.crop(vect)

# Plot the raster and vector
rast.plot(cmap="Blues")
vect.reproject(rast).plot(fc="none", ec="k", lw=2)

# %%
# We use the raster as a reference to match for rasterizing the proximity distances with :func:`~geoutils.Vector.proximity`. See :ref:`core-match-ref` for more details.

proximity = vect.proximity(rast)
proximity.plot(cmap="viridis")

# %%
# Proximity can also be computed to target pixels of a raster, or that of a mask

# Get mask of pixels within 30 of 200 infrared
import numpy as np

mask_200 = np.abs(rast - 200) < 30
mask_200.plot()

# %%
# Because a mask is :class:`bool`, no need to pass target pixels

proximity_mask = mask_200.proximity()
proximity_mask.plot(cmap="viridis")

# %%
# By default, proximity is computed using the georeference unit from a :class:`~geoutils.Raster`'s :attr:`~geoutils.Raster.res`, here **meters**. It can also
# be computed in pixels.

proximity_mask = mask_200.proximity(distance_unit="pixel")
proximity_mask.plot(cmap="viridis")
