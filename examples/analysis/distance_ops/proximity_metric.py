"""
Proximity to raster or vector
=============================

This example demonstrates the calculation of proximity distances to a raster or vector using :meth:`~geoutils.raster.base.RasterBase.proximity`.
"""

# %%
# We open an example raster, and a vector for which we select a single feature

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.open_raster(filename_rast)
vect = gu.open_vector(filename_vect)
vect = vect[vect["RGIId"] == "RGI60-15.10055"]
rast = rast.rst.crop(vect)

# Plot the raster and vector
rast.rst.plot(cmap="Blues")
vect.vct.reproject(rast).vct.plot(fc="none", ec="k", lw=2)

# %%
# We select the vector boundary, then use the raster as a reference to match for rasterizing the proximity distances
# with :meth:`~geoutils.vector.base.VectorBase.proximity`. See :ref:`core-match-ref` for more details.

boundary = vect.set_geometry(vect.boundary)
proximity = boundary.vct.proximity(rast.rst)
proximity.rst.plot(cmap="viridis")

# %%
# Proximity can also be computed to target pixels of a raster, or that of a mask

# Get mask of pixels within 30 of 200 infrared
import numpy as np

mask_200 = np.abs(rast - 200) < 30
mask_200.rst.plot()

# %%
# Because a mask is :class:`bool`, no need to pass target pixels

proximity_mask = mask_200.rst.proximity()
proximity_mask.rst.plot(cmap="viridis")

# %%
# By default, proximity is computed using the raster's georeferenced resolution, here **meters**. It can also
# be computed in pixels.

proximity_mask = mask_200.rst.proximity(distance_unit="pixel")
proximity_mask.rst.plot(cmap="viridis")
