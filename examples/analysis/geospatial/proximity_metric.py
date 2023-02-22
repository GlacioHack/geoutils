"""
Proximity to raster or vector
=============================

This example demonstrates the calculation of proximity distances to a raster or vector using :class:`~geoutils.Raster.proximity`.
"""
# sphinx_gallery_thumbnail_number = 2
# %%
# We open an example raster, and a vector for which we select a single feature
import geoutils as gu
rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))
vect = gu.Vector(vect.ds[vect.ds["RGIId"] == "RGI60-15.10055"])
rast.crop(vect)

# Plot the raster and vector
import matplotlib.pyplot as plt
ax = plt.gca()
rast.show(ax=ax, cmap="Blues")
vect.reproject(rast).ds.plot(ax=ax, fc='none', ec='k', lw=2)

# %%
# We use the raster as a reference to match for rasterizing the proximity distances with :func:`~geoutils.Vector.proximity`. See :ref:`core-match-ref` for more details.

proximity = vect.proximity(rast)
proximity.show(cmap="viridis")

# %%
# Proximity can also be computed to target pixels of a raster, or that of a mask

# Get mask of pixels within 30 of 200 infrared
import numpy as np
mask_200 = np.abs(rast - 200) < 30
mask_200.show()

# %%
# Because a mask is :class:`bool`, no need to pass target pixels

proximity_mask = mask_200.proximity()
proximity_mask.show(cmap="viridis")

# %%
# By default, proximity is computed using the georeference unit from a :class:`~geoutils.Raster`'s :attr:`~geoutils.Raster.res`, here **meters**. It can also
# be computed in pixels.

proximity_mask = mask_200.proximity(distance_unit='pixel')
proximity_mask.show(cmap="viridis")
