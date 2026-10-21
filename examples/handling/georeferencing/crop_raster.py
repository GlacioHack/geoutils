"""
Crop a raster
=============

This example demonstrates the cropping of a raster using :meth:`~geoutils.raster.base.RasterBase.crop`.
"""

# %%
# We open a raster and vector, and subset the latter.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.open_raster(filename_rast)
vect = gu.open_vector(filename_vect)
vect = vect[vect["RGIId"] == "RGI60-15.10055"]

# %%
# The first raster has larger extent and higher resolution than the vector.
rast.rst.info()
print(vect.vct.bbox)

# %%
# Let's plot the raster and vector.
rast.rst.plot(cmap="Purples")
vect.vct.plot(ref=rast, fc="none", ec="k", lw=2)

# %%
# **First option:** using the vector as a reference to match, we reproject the raster. We simply have to pass the vector
# as single argument to :meth:`~geoutils.raster.base.RasterBase.crop`. See :ref:`core-match-ref` for more details.

rast = rast.rst.crop(vect)

# %%
# Now the bounds should be the same as that of the vector (within the size of a pixel as the grid was not warped).
#
rast.rst.plot(ax="new", cmap="Purples")
vect.vct.plot(ref=rast, fc="none", ec="k", lw=2)

# %%
# **Second option:** we can pass other arguments to :meth:`~geoutils.raster.base.RasterBase.crop`, including another raster or a
# simple :class:`tuple` of bounds. For instance, we can re-crop the raster to be smaller than the vector.

rast = rast.rst.crop((rast.rst.bbox.left + 1000, rast.rst.bbox.bottom, rast.rst.bbox.right, rast.rst.bbox.top - 500))

rast.rst.plot(ax="new", cmap="Purples")
vect.vct.plot(ref=rast, fc="none", ec="k", lw=2)
