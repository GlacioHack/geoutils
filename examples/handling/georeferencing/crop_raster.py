"""
Crop a raster
=============

This example demonstrates the cropping of a raster using :func:`geoutils.Raster.crop`.
"""

# %%
# We open a raster and vector, and subset the latter.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)
vect = vect[vect["RGIId"] == "RGI60-15.10055"]

# %%
# The first raster has larger extent and higher resolution than the vector.
rast.info()
print(vect.bounds)

# %%
# Let's plot the raster and vector.
rast.plot(cmap="Purples")
vect.plot(ref_crs=rast, fc="none", ec="k", lw=2)

# %%
# **First option:** using the vector as a reference to match, we reproject the raster. We simply have to pass the :class:`~geoutils.Vector`
# as single argument to :func:`~geoutils.Raster.crop`. See :ref:`core-match-ref` for more details.

rast = rast.crop(vect)

# %%
# Now the bounds should be the same as that of the vector (within the size of a pixel as the grid was not warped).
#
# .. note::
#      By default, :func:`~geoutils.Raster.crop` is done in-place, replacing ``rast``. This behaviour can be modified by passing ``inplace=False``.
#

rast.plot(ax="new", cmap="Purples")
vect.plot(ref_crs=rast, fc="none", ec="k", lw=2)

# %%
# **Second option:** we can pass other ``crop_geom`` argument to :func:`~geoutils.Raster.crop`, including another :class:`~geoutils.Raster` or a
# simple :class:`tuple` of bounds. For instance, we can re-crop the raster to be smaller than the vector.

rast = rast.crop((rast.bounds.left + 1000, rast.bounds.bottom, rast.bounds.right, rast.bounds.top - 500))

rast.plot(ax="new", cmap="Purples")
vect.plot(ref_crs=rast, fc="none", ec="k", lw=2)
